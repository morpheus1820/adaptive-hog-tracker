#ifndef _MMCL_ODOM_H_
#define _MMCL_ODOM_H_

#include <math.h>

#include "mmcl_sensor.h"
#include "Map/map.h"
#include "pf/pf_pdf.h"

/**
  \brief Manage the odometry data

  The @p CMmcl_odom manages the odometry data and updates the particles position in accord with them.

  The @p actionModel function predicts the robot pose, every particle is moved in accord with the displacement and
  a random gaussian noise is added. The odometry configuration is stored in the @p mmcl.conf explained in the
  @p CMmcl documentation.

  Add an random noise to the motion data.

*/
class CMmcl_odom : public CMmcl_sensor
{
	public:
		CMmcl_odom(data_t idData, pf_vector_t pose, pf_matrix_t drift);

		pf_vector_t getPose();
		pf_vector_t getDelta();
		pf_pdf_gaussian_t* getPdf();

		void setDelta(pf_vector_t delta);
		void setDrift(pf_matrix_t drift);
		void setActionPdf(pf_pdf_gaussian_t *action_pdf);

		// prediction of the filter
		virtual bool updateAction(pf_t *pf);

	private:
		pf_vector_t m_pose;		// curret robot odometry pose
		pf_vector_t m_delta;	// robot movemet for the last odometry reading

		pf_matrix_t m_drift;	// drift of the robot odometry

		pf_pdf_gaussian_t *m_action_pdf;	// pdf of the odometry

		// model of the robot movment ( add gaussian noise )
		static void actionModel(CMmcl_odom *self, pf_sample_set_t* set);

};

#endif /*_MMCL_ODOM_H_*/
