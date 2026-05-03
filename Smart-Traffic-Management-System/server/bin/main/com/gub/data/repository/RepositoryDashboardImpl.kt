package com.gub.data.repository

import com.gub.data.service.dashboard.ServiceTrafficMeasure
import com.gub.data.service.dashboard.SystemOverviewService
import com.gub.domain.models.dashboard.ModelAiControl
import com.gub.domain.models.dashboard.ModelLiveTraffic
import com.gub.domain.models.dashboard.ModelSystemOverview
import com.gub.domain.repository.RepositoryDashboard

class RepositoryDashboardImpl(
    private val systemOverviewService: SystemOverviewService,
    private val serviceTrafficMeasure: ServiceTrafficMeasure
) : RepositoryDashboard {

    override suspend fun getSystemOverview(): ModelSystemOverview {
        return systemOverviewService.systemOverview()
    }

    override fun getLiveTrafficMetrics(): ModelLiveTraffic {
        return serviceTrafficMeasure.getLiveTrafficSummary()
    }

    override fun getAiControlSystem(): ModelAiControl {
        return ModelAiControl()
    }
}