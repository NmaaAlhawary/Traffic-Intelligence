package com.gub.domain.repository

import com.gub.domain.models.analytics.ModelTrafficVolume

interface RepositoryAnalytics {

    suspend fun getTrafficVolume(type: ModelTrafficVolume.TrafficVolumeType): ModelTrafficVolume
}