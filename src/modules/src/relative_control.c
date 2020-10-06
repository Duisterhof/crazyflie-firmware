#include "system.h"
#include "FreeRTOS.h"
#include "task.h"
#include "commander.h"
#include "relative_localization.h"
#include "num.h"
#include "param.h"
#include "debug.h"
#include <stdlib.h> // random
#include "lpsTwrTag.h" // UWBNum
#include "configblock.h"
#include "uart2.h"
#include "log.h"
#include <math.h>
#include "range.h"
#define USE_MONOCAM 0
#include "stabilizer_types.h"
#include "estimator_kalman.h"

// static float RAD2DEG = 57.29578049;
// static float critical_laser = 0.5; // no laser ranger should ever see lower than this
static float desired_laser = 2.5; // start correcting if a laser ranger sees smaller than this
static float desired_velocity = 0.6; // speed in m/s that we aim for
static bool isInit;
static bool onGround = true;
static bool keepFlying = false;
static setpoint_t setpoint;
static float_t relaVarInCtrl[NumUWB][STATE_DIM_rl];
static float_t inputVarInCtrl[NumUWB][STATE_DIM_rl];
static uint8_t selfID;
static float_t height;
static float old_vx = 0.0f;
static float old_vy = 0.0f;
static float rel_x = 0.0f;
static float rel_y = 0.0f;
static float heading_accumulator = 0.0f;

#define num_prev_readings 100
float previous_x[num_prev_readings];
float previous_y[num_prev_readings];

float x_min = 0.0f;
float x_max = 0.0f;
float y_min = 0.0f;
float y_max = 0.0f;
float x_range = 0.0f;
float y_range = 0.0f;

float lasers[4];
static float relaCtrl_p = 2.0f;
static float relaCtrl_i = 0.0001f;
static float relaCtrl_d = 0.01f;
static float wp_reached_thres = 0.2; // [m]
float heading = 0.0f;
float desired_heading = 0.0f;
// static float NDI_k = 2.0f;
static char c = 0; // monoCam
float search_range = 20.0; // search range in meters

float min_laser = 10.0f;
int laser_decision;

struct Point
{
    float x;
    float y;
};

// front, left, back right (ENU-based)
void getDistances(float* d) {
    *(d+0) = rangeGet(rangeFront)*0.001f;
    *(d+1) = rangeGet(rangeLeft)*0.001f;
    *(d+2) = rangeGet(rangeBack)*0.001f;
    *(d+3) = rangeGet(rangeRight)*0.001f;
}

float get_min(float* d)
{
  float min = d[0];
  if (d[1] < min)
  {
    min = d[1];
  }
  if (d[2] < min)
  {
    min = d[2];
  }
  if (d[3] < min)
  {
    min = d[3];
  }
  return min;
}

// function to determine if, given an obstacle, 'yawing' right or left is best
// d is given as front, left, back, right
// return 0 if 'yawing' positive is desired (ENU)
// return 1 if 'yawing' negative is desired (ENU)
// return 2 if no danger is present in the current movement direction
int decide_direction(float* d, float desired_heading )
{
  // making yaw positive
  // yaw is its desired direction
  if( desired_heading < 0)
  {
    desired_heading += (float)(M_PI)*2.f;
  }
  int quadrant = (int)(desired_heading/(float)(M_PI_2));
  int lower_idx = quadrant;
  int higher_idx = lower_idx+1;
  if ( higher_idx == 4)
  {
    higher_idx = 0;
  }
  // there's no danger in the moving direction
  if (d[lower_idx] > desired_laser && d[higher_idx] > desired_laser )
  {
    return 2;
  }
  // more space left, so move in postiive ENU
  else if (d[higher_idx] > d[lower_idx])
  {
    return 0;
  }
  // more space right, move in negative ENU
  else 
  {
    return 1;
  }

}

struct Point agent_pos,goal, random_point;

static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;
  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;
  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;
  setpoint->velocity_body = true;
  commanderSetSetpoint(setpoint, 3);
}


void flyVerticalInterpolated(float startz, float endz, float interpolate_time) {
    setpoint_t setpoint;
    int CMD_TIME = 100;   // new command ever CMD_TIME seconds
    int NSTEPS = (int) interpolate_time / CMD_TIME; 
    for (int i = 0; i < NSTEPS; i++) {
        float curr_height = startz + (endz - startz) * ((float) i / (float) NSTEPS);
        setHoverSetpoint(&setpoint, 0, 0, curr_height, 0); 
        commanderSetSetpoint(&setpoint, 3);
        vTaskDelay(100);
    }
}

bool check_collision(void)
{
  bool collision = false;
  float distance = 100.;
  for(int i = 0; i<NumUWB; i++)
  {
    if ( i != selfID)
    {
       distance = sqrtf(powf(relaVarInCtrl[i][STATE_rlX],2) + powf(relaVarInCtrl[i][STATE_rlY],2));
       if (distance < 0.5f)
       {
         collision = true;
       }       
    }
  }
  return collision;
}


float get_distance_points(struct Point p1, struct Point p2)
{
  return sqrtf(pow((p2.x-p1.x),2)+pow((p2.y-p1.y),2));
}

void relativeControlTask(void* arg)
{
  systemWaitStart();
  // height = (float)selfID*0.1f+0.2f;
  height = 1.5f;

  
  float vx = 0.f;
  float vy = 0.f;
  float wp_dist = 0.f;

  getDistances(lasers);
  while(1) {
    vTaskDelay(10);
    getDistances(lasers);
    keepFlying = command_share(selfID, keepFlying);
    
    point_t state;
    estimatorKalmanGetEstimatedPos(&state); //read agent state from kalman filter
    agent_pos.x = state.x;
    agent_pos.y = state.y;


    
    // float prev_x[pos_hist];
    // append_array(prev_x,5.0,pos_hist);
    // avg_x = get_average(prev_x,pos_hist);
    // float prev_y[pos_hist];
    // DEBUG_PRINT("%d %d \n", keepFlying,relativeInfoRead((float_t *)relaVarInCtrl, (float_t *)inputVarInCtrl) );
    // if(relativeInfoRead((float_t *)relaVarInCtrl, (float_t *)inputVarInCtrl) && keepFlying){
    if(keepFlying){
      // take off
      if(onGround){
        for (int i=0; i<5; i++) {
          setHoverSetpoint(&setpoint, 0, 0, 0.3f, 0);
          vTaskDelay(M2T(100));
        }

        onGround = false;
      }

      // control loop
      // setHoverSetpoint(&setpoint, 0, 0, height, 0); // hover
      point_t state;
      random_point.x = (rand()/(float)RAND_MAX)*search_range-0.5f*search_range;
      random_point.y = (rand()/(float)RAND_MAX)*search_range-0.5f*search_range;
      goal = random_point;
      estimatorKalmanGetEstimatedPos(&state); //read agent state from kalman filter
      agent_pos.x = state.x;
      agent_pos.y = state.y;
      wp_dist = get_distance_points(agent_pos,random_point);


    // patter detection
    for (int i = num_prev_readings-1 ; i >0; i--)
    {
      previous_x[i] = previous_x[i-1];
      previous_y[i] = previous_y[i-1];
    }
    previous_x[0] = agent_pos.x;
    previous_y[0] = agent_pos.y;

    x_min = 1000.0f;
    y_min = 1000.0f;
    x_max = -1000.0f;
    y_max = -1000.0f;
    for (int i = 0; i < num_prev_readings ; i++)
    {
      if (previous_x[i] > x_max)
      {
        x_max = previous_x[i];
      }
      if (previous_x[i] < x_min)
      {
        x_min = previous_x[i];
      }

      if (previous_y[i] > y_max)
      {
        y_max = previous_y[i];
      }
      if (previous_y[i] < y_min)
      {
        y_min = previous_y[i];
      }
    }

    x_range = (x_max-x_min);
    y_range = (y_max-y_min);
    



        // float accumulator_obs_avoidance = 0.0f;

        for (int i = 0; i< 700; i++) //time before time-out
        {
          if (wp_dist >wp_reached_thres)
          {
          estimatorKalmanGetEstimatedPos(&state); //read agent state from kalman filter
          relativeInfoRead((float_t *)relaVarInCtrl, (float_t *)inputVarInCtrl); //get relative state from other agents
          getDistances(lasers); // get laser ranger readings, order: front, left, back, right
          // min_laser = get_min(lasers); // get minimum value of laser rangers
          agent_pos.x = state.x;
          agent_pos.y = state.y;
          rel_x = goal.x-agent_pos.x;
          rel_y = goal.y-agent_pos.y;
          wp_dist = get_distance_points(agent_pos,random_point);



          if ( wp_dist > wp_reached_thres)
          {
          heading = atan2f((goal.y-agent_pos.y),(goal.x-agent_pos.x)) + heading_accumulator; // heading in deg (ENU) to desired waypoint
          desired_heading = atan2f((goal.y-agent_pos.y),(goal.x-agent_pos.x));
          vx = desired_velocity*cosf(heading);
          vy = desired_velocity*sinf(heading);          

          // add repulsive forces from lasers
          for (int i = 0; i < 4; i++)
          {
            if (lasers[i] < desired_laser)
            {
              float laser_heading = (float)(i)*(float)(M_PI_2);
              float laser_repulse_heading = laser_heading + (float)(M_PI);
              vx += cosf(laser_repulse_heading)*2.0f*powf((desired_laser-lasers[i]),2);
              vy += sinf(laser_repulse_heading)*2.0f*powf((desired_laser-lasers[i]),2);
            }
          }
          
          //scale back to desired velocity
          float vector_size = sqrtf(powf(vx,2)+powf(vy,2));
          vx = vx/vector_size*desired_velocity;
          vy = vy/vector_size*desired_velocity;

          vx = 0.1f*old_vx + 0.9f * vx;
          vy = 0.1f*old_vy + 0.9f * vy;

          old_vx = vx;
          old_vy = vy;

          setHoverSetpoint(&setpoint,vx,vy,height,0);

          }
          else
          {
            setHoverSetpoint(&setpoint,0,0,height,0);
          }
          
          vTaskDelay(M2T(100));
        }
      
        }
    }
    else{
      // landing procedure
      if(!onGround){
        flyVerticalInterpolated(height,0.0f,6000.0f);
        onGround = true;
      } 
    }
  }
}

void relativeControlInit(void)
{
  if (isInit)
    return;
  selfID = (uint8_t)(((configblockGetRadioAddress()) & 0x000000000f) - 5);
#if USE_MONOCAM
  if(selfID==0)
    uart2Init(115200); // only CF0 has monoCam and usart comm
#endif
  xTaskCreate(relativeControlTask,"relative_Control",configMINIMAL_STACK_SIZE, NULL,3,NULL );
  isInit = true;
}

PARAM_GROUP_START(relative_ctrl)
PARAM_ADD(PARAM_UINT8, keepFlying, &keepFlying)
PARAM_ADD(PARAM_FLOAT, relaCtrl_p, &relaCtrl_p)
PARAM_ADD(PARAM_FLOAT, relaCtrl_i, &relaCtrl_i)
PARAM_ADD(PARAM_FLOAT, relaCtrl_d, &relaCtrl_d)

PARAM_GROUP_STOP(relative_ctrl)

LOG_GROUP_START(mono_cam)
LOG_ADD(LOG_UINT8, charCam, &c)
LOG_GROUP_STOP(mono_cam)

LOG_GROUP_START(lasers)
LOG_ADD(LOG_FLOAT,front,&lasers[0])
LOG_ADD(LOG_FLOAT,left,&lasers[1])
LOG_ADD(LOG_FLOAT,back,&lasers[2])
LOG_ADD(LOG_FLOAT,right,&lasers[3])
LOG_ADD(LOG_FLOAT,heading,&heading)
LOG_ADD(LOG_FLOAT,rel_x,&rel_x)
LOG_ADD(LOG_FLOAT,rel_y,&rel_y)
LOG_ADD(LOG_FLOAT,x_range,&x_range)
LOG_ADD(LOG_FLOAT,y_range,&y_range)
LOG_GROUP_STOP(lasers)