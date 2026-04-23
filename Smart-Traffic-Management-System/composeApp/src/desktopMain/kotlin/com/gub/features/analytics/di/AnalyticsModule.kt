package com.gub.features.analytics.di

import com.gub.di.NetworkModule.provideHttpClient
import com.gub.features.analytics.data.remote.AnalyticsApisImpl
import com.gub.features.analytics.data.repository.RepositoryAnalyticsImpl
import com.gub.features.analytics.domain.usecase.UseCaseTrafficVolume

object AnalyticsModule {

    private val analyticsApis by lazy {
        AnalyticsApisImpl(provideHttpClient())
    }

    private val repositoryAnalytics by lazy {
        RepositoryAnalyticsImpl(analyticsApis)
    }

    val useCaseTrafficVolume by lazy {
        UseCaseTrafficVolume(repositoryAnalytics)
    }
}