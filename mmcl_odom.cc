#include "mmcl_odom.h"
#include <iostream>

/**
Create CMmcl_odom class

@param  idData data type of the sensor
@param	pose odometry current pose
@param  drift matrix of the extimed robot drift
*/
CMmcl_odom::CMmcl_odom(data_t idData, pf_vector_t pose, pf_matrix_t drift)
	: CMmcl_sensor(idData)
{
	m_pose = pose;
	m_delta = pf_vector_zero();

	m_drift = drift;
}

/**
 * Get the odometry pose
 * @return odometry pose
 */
pf_vector_t CMmcl_odom::getPose()
{
	return m_pose;
}

/**
 *	Get the delta movement of the robot
 *	@return vector with dx, dy and dtheta (m, m, rad)
 */
pf_vector_t CMmcl_odom::getDelta()
{
	return m_delta;
}

/**
 *	Get the pdf function
 *	@return the pdf function
 */
pf_pdf_gaussian_t* CMmcl_odom::getPdf()
{
	return m_action_pdf;
}

/**
 * Set delta robot movement between two following odometry data
 * This displacement is used to predict the robot pose.
 * @param delta dx, dy and dtheta (m, m, rad)
 */
void CMmcl_odom::setDelta(pf_vector_t delta)
{
	m_delta = delta;
}

/**
 * Set the robot extime drift matrix
 * @param drift matrix (3x3)
 */
void CMmcl_odom::setDrift(pf_matrix_t drift)
{
	m_drift = drift;
}

/**
 * Set the pdf action for update the particles pose
 * @param action_pdf pdf gaussian function
 */
void CMmcl_odom::setActionPdf(pf_pdf_gaussian_t *action_pdf)
{
	m_action_pdf = action_pdf;
}

/**
 * Update the particle filter state (update the particles positions)
 * @param pf particle filter
 * @return true if the method is implemented
 */
bool CMmcl_odom::updateAction(pf_t *pf)
{
	pf_vector_t x;
  	pf_matrix_t cx;
  	double ux, uy, ua;

  	/*
  	printf("odom: %f %f %f : %f %f %f\n",
         ndata->pose.v[0], ndata->pose.v[1], ndata->pose.v[2],
         ndata->delta.v[0], ndata->delta.v[1], ndata->delta.v[2]);
  	*/

  	// See how far the robot has moved
  	x = m_delta;

  	// Odometric drift model
  	// This could probably be improved
  	ux = m_drift.m[0][0] * x.v[0];
  	uy = m_drift.m[1][1] * x.v[1];
  	ua = m_drift.m[2][0] * fabs(x.v[0])
    	+ m_drift.m[2][1] * fabs(x.v[1])
    	+ m_drift.m[2][2] * fabs(x.v[2]);

  	cx = pf_matrix_zero();
  	cx.m[0][0] = ux * ux;
  	cx.m[1][1] = uy * uy;
  	cx.m[2][2] = ua * ua;

//  	printf("x = %f %f %f\n", x.v[0], x.v[1], x.v[2]);

  	// Create a pdf with suitable characterisitics
  	m_action_pdf = pf_pdf_gaussian_alloc(x, cx);

  	// Update the filter
  	pf_update_action_update_cluster(pf, (pf_action_model_fn_t) actionModel, this);

  	// Delete the pdf
  	pf_pdf_gaussian_free(m_action_pdf);

  	return true;
}

/**
 * Action model of the robot (static)
 * @param self pointer to the class
 * @param set sample set to update
 */
void CMmcl_odom::actionModel(CMmcl_odom *self, pf_sample_set_t* set)
{
	int i;
  	pf_vector_t z;
  	pf_sample_t *sample;
//  	map_t *map = self->getMap();

  	// Compute the new sample poses
  	for (i = 0; i < set->sample_count; i++)
  	{
    	sample = set->samples + i;

		z = pf_pdf_gaussian_sample(self->getPdf());
		sample->pose = pf_vector_coord_add(z, sample->pose);

//		int ix = (int)MAP_GXWX(map, sample->pose.v[0]);
//		int iy = (int)MAP_GYWY(map, sample->pose.v[1]);
//
//		if( !MAP_VALID(map, ix, iy))
//		{
//			double xm = (map->size_x/2);
//			double ym = (map->size_y/2);
//
//			if( sample->pose.v[0] < -xm )
//				sample->pose.v[0] = -xm;
//
//			if( sample->pose.v[0] > xm )
//				sample->pose.v[0] = xm;
//
//			if( sample->pose.v[1] < -ym )
//				sample->pose.v[1] = -ym;
//
//			if( sample->pose.v[1] > ym )
//				sample->pose.v[1] = ym;
//
//		}

//		if( map->cells[MAP_INDEX(map, ix, iy)].occ_state != -1 )
//		{
//			sample->weight /= 100;
//		}


//		if( !correct )
//			std::cout << "Non corretto "<<sample->pose.v[0] << " " << sample->pose.v[1] << " " << ix << " " << iy << std::endl;


//    	sample->weight = 1.0 / set->sample_count;
  	}
}

//void CMmcl_odom::actionModel(CMmcl_odom *self, pf_sample_set_t* set)
//{
//	int i;
//  	pf_vector_t z;
//  	pf_sample_t *sample;
//
//  	// Compute the new sample poses
//  	for (i = 0; i < set->sample_count; i++)
//  	{
//    	sample = set->samples + i;
//    	z = pf_pdf_gaussian_sample(self->getPdf());
//    	sample->pose = pf_vector_coord_add(z, sample->pose);
//    	sample->weight = 1.0 / set->sample_count;
//  	}
//}
