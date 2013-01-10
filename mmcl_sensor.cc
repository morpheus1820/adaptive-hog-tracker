#include "mmcl_sensor.h"

CMmcl_sensor::CMmcl_sensor(data_t idData)
{
	m_idData = idData;
}

CMmcl_sensor::~CMmcl_sensor()
{}

CMmcl_sensor::data_t CMmcl_sensor::getIdData()
{
	return m_idData;
}

bool CMmcl_sensor::updateAction(pf_t *pf)
{
	return false;
}
