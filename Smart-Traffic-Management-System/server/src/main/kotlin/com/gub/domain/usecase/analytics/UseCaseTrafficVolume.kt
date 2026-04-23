package com.gub.domain.usecase.analytics

import com.gub.domain.models.analytics.ModelTrafficVolume
import com.gub.domain.repository.RepositoryAnalytics

class UseCaseTrafficVolume(
    private val repositoryAnalytics: RepositoryAnalytics
) {

    suspend operator fun invoke(type: ModelTrafficVolume.TrafficVolumeType) : ModelTrafficVolume {
        return repositoryAnalytics.getTrafficVolume(type)
    }
}