#ifndef _MMCL_MUTUALHYP_H_
#define _MMCL_MUTUALHYP_H_

#include <math.h>
#include <vector>

#include "mmcl_sensor.h"
#include "Map/map.h"
#include "pf/pf_pdf.h"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
/**
  \brief manages the position received by the other robot

  \p CMmcl_mutualHyp receives the positions sent by another robot and stores them in the data queue.
  At the first particle filter update, for each position some particles are added at the particle.
  The number of particles added depends by the number of the received positions and the current number of
  particles in the filter.

  The particles are added by \p pf_update_resample_hyps with a bivariate gaussian function for
  (x,y) with the hypothesis covariance. Instead the \f$\theta\f$ is uniformly added because
  no information is available by the other robot.

 */
class CMmcl_hog : public CMmcl_sensor
{

	public:
		CMmcl_hog(std::vector<hyp_t *> hyps,cv::Mat & hm);
		CMmcl_hog(cv::Mat & hm);
		~CMmcl_hog();
		pf_pdf_gaussian_t* getPdf();

		int getNRcvHyp();
		int getNRcvParticle();
		hyp_t getHyp(int i);


		typedef enum {
			HYPS,
			MAP


		} mode;
		/**
		 * Sender robot conditions to update the receiver one particles filter
		 */
		typedef struct mutualHyps
		{
			/**
			 * sender id
			 */
			int id;
			/**
			 * sender localization state
			 */
			int state;
			/**
			 * sender particles number
			 */
			int nParticle;
			/**
			 * sender hypotheses number
			 */
			int nHyps;

		}mutualHyps_t;

		int setHyps(std::vector<hyp_t *> hyps);

		void setActionPdf(pf_pdf_gaussian_t *action_pdf);

		// prediction of the filter
		virtual bool updateAction(pf_t *pf);

		// for hm weighting
		void setHeatMap(const Mat & hm);
		void setMap(map_t *map);
	private:
		int m_senderId;
		int m_nRcvHyp;
		bool m_send_lw_hyps;
		double	m_w_thsld;
		int m_nRcvParticle;
		std::vector<hyp_t* > m_hyps;	//list of the hypotesis

		pf_pdf_gaussian_t *m_action_pdf;	// pdf of the odometry

		// compute the weight of every particle
		static double sensorModelMap(CMmcl_hog *self, pf_sample_set_t* set);
		static double sensorModelHyps(CMmcl_hog *self, pf_sample_set_t* set);

		map_t *m_map;
		Mat & m_heatMap;
		mode m_mode;
};

#endif /*_MMCL_MUTUALHYP_H_*/
