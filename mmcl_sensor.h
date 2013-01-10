/*
  @file    mmcl_sensor.h
  @author  Federico Tibaldi, federico.tibaldi@polito.it
  @version 1.0
*/

#ifndef _MMCL_SENSOR_H_
#define _MMCL_SENSOR_H_

#include "pf/pf.h"

/**
  \brief Root class for the sensor

  Root class for the sensor (odometry, range, etc.)

  It implements an interface for the classes that manage the sensor readings and
  the update of the particle filter.
*/
class CMmcl_sensor
{
	public:
		/**
		Type of managed sensor type.
		*/

		typedef enum{LASER_DATA,  /**< laser data */
						ODO_DATA,  /**< odometry data */
						SONAR_DATA,  /**< sonar data */
						MUTUAL_DATA,  /**< mutual localization data */
						BADHYP_DATA,  /**< bad hypothesis data */
						IMU_DATA, /**< IMU data */
						WIFI_DATA,
						RFID_DATA,
						OTHER_DATA /**< NOT valid data */
					} data_t;

		/**
	    Create CMmcl_sensor class

	    @param  idData data type of the sensor

		*/
		CMmcl_sensor(data_t idData);

		/**
	    Destroy CMmcl_sensor class
	    */
		virtual ~CMmcl_sensor();

		/**
		Get the sensor type id
		@return the type id
		*/
		data_t getIdData();

		/**
		Pure virtual method, every derived class has to implement it.
		It updates the particle filter in accord to the sensor type and
		data readings.

		@param pf the particle filter pointer

		@return true if the method is implemented
		*/
		virtual bool updateAction(pf_t *pf);

	private:
		/** type of data menaged by the class */
		data_t m_idData;
};

#endif /*_MMCL_SENSOR_H_*/
