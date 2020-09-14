#define DEBUG_MODULE "HelloDeck"
/* TFMicro Test Script: Tests to see the delay incurred by TF-Micro and
how fast the chip can process these neural networks */
#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements
#include "log.h"
#include "FreeRTOS.h"
#include "task.h"
#include "log.h"
#include "debug.h"
#include "sysload.h"

#include "deck_analog.h"
// uTensor related machine 
float R_s;

static void readGas()
{
  systemWaitStart();
  DEBUG_PRINT("GAS READ");
  vTaskDelay(M2T(500));
  float voltage = 0.0;
  while(1)
  {
    float voltage = analogReadVoltage(DECK_GPIO_RX2);
    // R_s = (3.0/voltage-1.0)*69;
    R_s = voltage;
    vTaskDelay(M2T(150));
  }
}

static void gasInit()
{
  xTaskCreate(readGas, "read_gas",
		1000 /* Stack size in terms of WORDS (usually 4 bytes) */,
		NULL, /*priority*/3, NULL);

  DEBUG_PRINT("Hello Crazyflie 2.0 deck world!\n");
}

static bool gasTest()
{
  DEBUG_PRINT("Hello test passed!\n");
  return true;
}

static const DeckDriver gasDriver = {
  .name = "gasDeck",
  .init = gasInit,
  .test = gasTest,
};

DECK_DRIVER(gasDriver);

LOG_GROUP_START(gas)
LOG_ADD(LOG_FLOAT,R,&R_s)
LOG_GROUP_STOP(gas)
