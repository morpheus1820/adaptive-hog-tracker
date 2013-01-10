#include "mmcl_hog.h"
#include <iostream>

CMmcl_hog::CMmcl_hog(std::vector<hyp_t *> hyps, cv::Mat & hm) : CMmcl_sensor(MUTUAL_DATA), m_heatMap(hm)
{
	m_send_lw_hyps = false;
	m_w_thsld = 1.0;
	m_nRcvHyp = 0;

	m_hyps = hyps;
	m_nRcvHyp=hyps.size();

	m_mode=HYPS;

}

CMmcl_hog::CMmcl_hog(cv::Mat & hm) : CMmcl_sensor(MUTUAL_DATA), m_heatMap(hm)
{
	m_mode=MAP;
}

CMmcl_hog::~CMmcl_hog()
{
	m_hyps.clear();
}

int CMmcl_hog::getNRcvHyp()
{
	return m_nRcvHyp;
}

int CMmcl_hog::getNRcvParticle()
{
	return m_nRcvParticle;
}

hyp_t CMmcl_hog::getHyp(int i)
{
	return *m_hyps[i];
}

int CMmcl_hog::setHyps(std::vector<hyp_t *> hyps)
{
	m_hyps = hyps;
	m_nRcvHyp=hyps.size();

	return hyps.size();
}

bool CMmcl_hog::updateAction(pf_t *pf)
{
  	// Apply the right sensor model
	if(m_mode==MAP)
		pf->sumSquareWeights = pf_update_sensor(pf, (pf_sensor_model_fn_t) sensorModelMap, this);
	if(m_mode==HYPS)
		pf->sumSquareWeights = pf_update_sensor(pf, (pf_sensor_model_fn_t) sensorModelHyps, this);
	return true;
}


void CMmcl_hog::setHeatMap(const Mat & hm)
{
	m_heatMap=hm.clone();
}

void CMmcl_hog::setMap(map_t *map)
{
	m_map=map;
}


double CMmcl_hog::sensorModelMap(CMmcl_hog *self, pf_sample_set_t* set )
{
	using namespace cv;
	std::cout << "Update map"<<std::endl;
  	double total_weight;
  	pf_sample_t *sample;
  	pf_vector_t pose;


  	// delta between the sample position and the hypothesis position
  	pf_vector_t delta;
  	// weitght sum of the distance between particle and hypothesis
	double dist;

  	double pz;
  	double p;

  	double zx, zy, z, ro;

  	total_weight = 0.0;

  	// Compute the sample weights
  	for (int j = 0; j < set->sample_count; j++)
  	{
   		sample = set->samples + j;
    	pose = sample->pose;

		p = 1.0;
		dist = 0;

		// HEATMAP, difference obs-true, as with wifimaps
		int ix = MAP_GXWX(self->m_map, sample->pose.v[0]);
		int iy = MAP_GYWY(self->m_map, -sample->pose.v[1]);

		unsigned int value=self->m_heatMap.at<unsigned char>(iy,ix);

		z=(255.0-(float)value);

		// aggiungo bad range prob
		double bad_range=0.3;
		double c=0.01; //covariance
		pz=exp(-(z*z)/(2.0*c*c));
		double pz_br = bad_range + (1.0-bad_range)*pz;

		p *= pz_br;


//		std::cout << "p " << j << " " << dist << std::endl;

    	sample->weight *= p;
    	total_weight += sample->weight;
  	}

  	for (int j = 0; j < set->sample_count; j++)
  	{
   		sample = set->samples + j;
    	sample->weight /= total_weight;
  	}

  	return(total_weight);
}

// sensor model based on hog detections hyps
double CMmcl_hog::sensorModelHyps(CMmcl_hog *self, pf_sample_set_t* set )
{
	using namespace cv;

	printf("Update hyps\n");
  	double total_weight;
  	int nHyp = self->getNRcvHyp();
  	pf_sample_t *sample;
  	pf_vector_t pose;


  	// delta between the sample position and the hypothesis position
  	pf_vector_t delta;
  	// weitght sum of the distance between particle and hypothesis
	double dist;

  	double pz;
  	double p;

  	double zx, zy, z, ro;

  	total_weight = 0.0;

  	// Compute the sample weights
  	for (int j = 0; j < set->sample_count; j++)
  	{
   		sample = set->samples + j;
    	pose = sample->pose;

		p = 1.0;
		dist = 0;

		for( int i = 0; i < nHyp; i++)
		{

			delta = pf_vector_zero();
			hyp_t tmp = self->getHyp(i);
			delta = pf_vector_sub(pose, tmp.pf_pose_mean);
			dist += (delta.v[0] * delta.v[0] + delta.v[1] * delta.v[1]) * tmp.weight;
//			std::cout << "Update su ipotesi "<<tmp.pf_pose_mean.v[0]<<" "<<tmp.pf_pose_mean.v[1]<<std::endl;
			// (x - mu)^2 / sigma
			zx = (delta.v[0] * delta.v[0]) / (tmp.pf_pose_cov.m[0][0]);
			zy = (delta.v[1] * delta.v[1]) / (tmp.pf_pose_cov.m[1][1]);

			// correlation ro between x, y
			ro = tmp.pf_pose_cov.m[0][1] / (tmp.pf_pose_cov.m[0][0] * tmp.pf_pose_cov.m[1][1]);


			// circular weight, as with hyps
			z = zx + zy - 2 * ro * delta.v[0] * delta.v[1] / (tmp.pf_pose_cov.m[0][0] * tmp.pf_pose_cov.m[1][1]);



			//pz = exp(-z/(2*(1-ro)*(1-ro)))/( 2*M_PI*tmp.pf_pose_cov.m[0][0]*tmp.pf_pose_cov.m[1][1]*sqrt(1-ro*ro));



			// aggiungo bad range prob
			double bad_range=0.3;
			pz=exp(-(z*z)/(2.0*(1.0-ro)*(1.0-ro)));
			double pz_br = bad_range + (1.0-bad_range)*pz;

			p *= pz_br;

//			std::cout << delta.v[0] << " " << delta.v[1] << " " << pz << std::endl;

		}

//		std::cout << "p " << j << " " << dist << std::endl;

    	sample->weight *= p;
    	total_weight += sample->weight;
  	}

  	for (int j = 0; j < set->sample_count; j++)
  	{
   		sample = set->samples + j;
    	sample->weight /= total_weight;
  	}

  	return(total_weight);
}
