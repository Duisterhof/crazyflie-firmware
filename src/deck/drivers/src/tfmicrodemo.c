/* TFMicro Test Script: Tests to see the delay incurred by TF-Micro and
how fast the chip can process these neural networks */
#include <time.h>
#include <stdlib.h>

#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements
#include "log.h"

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"
#include "machinelearning.h"
#include "sysload.h"
#include "sequencelib.h"
#include "sensor.h"
#include "task.h"
#include "timers.h"
#include "math.h"
#include "tfmicrodemo.h"
#include "estimator_kalman.h"
#include "stabilizer_types.h"
// uTensor related machine learning

#define SUBTRACT_VAL 60
#define STATE_LEN 5
#define NUM_STATES 4
#define YAW_INCR 5
//#define SENS_MIN 35000
#define SENS_MIN 0
#define SENS_MAX 65000
#define TRUE 1
#define FALSE 0
#define GOAL_THRES 245
#define GOAL_THRES_COUNT 3
#define DIST_MIN 90
#define RAND_ACTION_RATE 30
#define RAD2DEG 57.29578049

struct Point agent_pos,goal;

float yaw_incr(float yaw){
    float yaw_out = yaw + YAW_INCR;
    if(yaw_out>=180){
        yaw_out -= 360;
    }
    return yaw_out;
}


float yaw_decr(float yaw){
    float yaw_out = yaw - YAW_INCR;
    if(yaw_out<=-180){
        yaw_out += 360;
    }
    return yaw_out;
}

int argmax_float(float* array, int size){
    float max = array[0];
    int max_ind = 0;
    for (int i = 1; i< size; i++){
        if (array[i]>max){
            max = array[i];
            max_ind = i;
        }
    }
    return max_ind;
}


static void check_multiranger_online() {
	DEBUG_PRINT("Checking if multiranger ToF sensors online...\n");

	distances d;
	for (int j = 0; j < 10; j++) {
		getDistances(&d);
		vTaskDelay(M2T(100));
	}
	if (d.left == 0 && d.right == 0) {
		// most likely the ranger deck isn't attached correctly
		DEBUG_PRINT("Most likely ranger deck not attached correctly\n");
		for (;;) {
			vTaskDelay(M2T(1000));
		} 
	}
}



// get distance between two points
float get_distance(struct Point p1, struct Point p2)
{
    return sqrtf(powf((p2.x-p1.x),2)+powf((p2.y-p1.y),2));
}

float get_heading(struct Point p1, struct Point p2)
{
    return(RAD2DEG*atan2f((p2.y-p1.y),(p2.x-p1.x)));
}

static void tfMicroDemoTask()
{
    int command = 0;
    float yaw = 0;

    int x_range = 6; // the range in which we explore
    int y_range = 6; // the range in which we explore
	static setpoint_t setpoint;
	systemWaitStart();

    // time parameters
    

	float ESCAPE_SPEED = 1.0;
    float goal_dist_thres = 0.5;
    float HOVER_HEIGHT = 1.0;
    float rotate_threshold = 1.0;
    float update_time = 10.0; //set a new goal every 10 seconds
    // Start in the air before doing ML
    flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 6000.0f);
    vTaskDelay(M2T(500));
    distances d;
    getDistances(&d);

    float front_sensor = d.front*0.001;
    float right_sensor = d.right*0.001;
    float left_sensor = d.left*0.001;
    float back_sensor = d.back*0.001;

    bool wall_following = false;
    srand(time(NULL));
    int r = rand();
    int r2 = rand();
    uint32_t tick_start = xTaskGetTickCount();
    
    goal.x = (float)(r%(x_range*10))/(10.)-(float)(x_range)*0.5;
    goal.y = (float)(r2%(y_range*10))/(10.)-(float)(y_range)*0.5;
    point_t state;
    
    for (int j = 0; j < 10000; j++) {
        // state = get_state();
        estimatorKalmanGetEstimatedPos(&state);
        agent_pos.x = state.x;
        agent_pos.y = state.y;
        getDistances(&d);
        float local_time = (float)(xTaskGetTickCount()-tick_start )/(1000.);

        // safety statement -- kill drone when hand is over < 20 cm
        // if(d.up/10 < 20)
        // {
        //     flyVerticalInterpolated(HOVER_HEIGHT, 0.1f, 1000.0f);
	    //     for (;;) { vTaskDelay(M2T(1000)); }
        //     break;
        // }

        if (local_time > update_time || get_distance(agent_pos,goal) < goal_dist_thres)
        // if (local_time > update_time )
        {
            tick_start = xTaskGetTickCount();
            r = rand();
            r2 = rand();
            goal.x = (float)(r%(x_range*10))/(10.)-(float)(x_range)*0.5;
            goal.y = (float)(r2%(y_range*10))/(10.)-(float)(y_range)*0.5;
            DEBUG_PRINT("%f %f",goal.x,goal.y);
            yaw = get_heading(agent_pos,goal);
        }
        
        front_sensor = d.front*0.001;    // used for obs avoidance
        right_sensor = d.right*0.001;
        left_sensor = d.left*0.001;
        back_sensor = d.back*0.001;


        if (front_sensor < rotate_threshold)
        {
            wall_following = true;
            if (right_sensor >left_sensor)
            {
                command = 2;
            }
            else
            {
                command = 1;
            }
        }
        else
        {
            if (wall_following == true)
            {
                if( front_sensor > rotate_threshold && right_sensor > rotate_threshold && left_sensor > rotate_threshold)
                {
                    wall_following = false;
                    command = 0;
                }
                else
                {
                    command = 3;
                }  
            }
            else
            {
                command = 0;            
            } 
        }
        
        switch (command) {
          // fly forward
          case 0:
              yaw = get_heading(agent_pos,goal);
              DEBUG_PRINT("%f \n",yaw);
              setHoverSetpoint(&setpoint, ESCAPE_SPEED, 0, HOVER_HEIGHT, yaw);
              commanderSetSetpoint(&setpoint, 3);
              vTaskDelay(M2T(100));
              break;

          // rotate left
          case 1:
                yaw = yaw_incr(yaw);
                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT,yaw);
                commanderSetSetpoint(&setpoint, 3);
                vTaskDelay(M2T(100));    
              break;

            // rotate right
            case 2:
                yaw = yaw_decr(yaw);
                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT,yaw);
                commanderSetSetpoint(&setpoint, 3);
                vTaskDelay(M2T(100));    
               break;
            
            case 3:
              setHoverSetpoint(&setpoint, ESCAPE_SPEED, 0, HOVER_HEIGHT, yaw);
              commanderSetSetpoint(&setpoint, 3);
              vTaskDelay(M2T(100));
              break;
      }
}

    flyVerticalInterpolated(HOVER_HEIGHT, 0.1f, 1000.0f);
	for (;;) { vTaskDelay(M2T(1000)); }
}

static void init() {
	xTaskCreate(tfMicroDemoTask, "tfMicroDemoTask",
		4500 /* Stack size in terms of WORDS (usually 4 bytes) */,
		NULL, /*priority*/3, NULL);
}

static bool test() {
	return true;
}

const DeckDriver tf_micro_demo = {
	.vid = 0,
	.pid = 0,
	.name = "tfMicroDemo",

	.usedGpio = 0,  // FIXME: set the used pins

	.init = init,
	.test = test,
};

DECK_DRIVER(tf_micro_demo);

LOG_GROUP_START(gas_log)
LOG_ADD(LOG_FLOAT, goal_x, &goal.x)
LOG_ADD(LOG_FLOAT, goal_y, &goal.y)
LOG_ADD(LOG_FLOAT, agent_x, &agent_pos.x)
LOG_ADD(LOG_FLOAT, agent_y, &agent_pos.y)
LOG_GROUP_STOP(gas_log)