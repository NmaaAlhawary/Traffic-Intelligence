package com.gub.features.analytics.data.repository

import com.gub.domain.models.analytics.ModelTrafficVolume
import com.gub.features.analytics.data.remote.AnalyticsApis
import com.gub.features.analytics.domain.repository.RepositoryAnalytics

class RepositoryAnalyticsImpl(
    private val analyticsApisImpl: AnalyticsApis
) : RepositoryAnalytics {

    override suspend fun getTrafficVolume(type: ModelTrafficVolume.TrafficVolumeType): ModelTrafficVolume {
        return analyticsApisImpl.getTrafficVolume(type)
    }
}