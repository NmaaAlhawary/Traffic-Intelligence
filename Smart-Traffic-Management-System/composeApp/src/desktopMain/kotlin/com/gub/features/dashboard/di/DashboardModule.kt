package com.gub.features.dashboard.di

import com.gub.di.NetworkModule.provideHttpClient
import com.gub.features.dashboard.data.remote.SystemOverviewApi
import com.gub.features.dashboard.data.remote.SystemOverviewApiImpl
import com.gub.features.dashboard.data.repository.SystemOverviewRepositoryImpl
import com.gub.features.dashboard.domain.repository.SystemOverviewRepository
import com.gub.features.dashboard.domain.usecase.GetSystemOverviewUseCase
import com.gub.features.dashboard.domain.usecase.UpdateSystemMetricsUseCase
import com.gub.features.monitoring.data.network.LiveSignalApi
import com.gub.features.monitoring.data.network.LiveSignalApiImpl
import com.gub.features.monitoring.data.repository.RepositoryLiveSignalImpl
import com.gub.features.monitoring.domain.repository.RepositoryLiveSignal
import com.gub.features.monitoring.domain.usecase.UseCaseLiveSignal

object DashboardModule {

    private val systemOverviewApi: SystemOverviewApi by lazy {
        SystemOverviewApiImpl(provideHttpClient())
    }

    private val systemOverviewRepository: SystemOverviewRepository by lazy {
        SystemOverviewRepositoryImpl(systemOverviewApi)
    }

    val getSystemOverviewUseCase by lazy { GetSystemOverviewUseCase(systemOverviewRepository) }

    val updateSystemMetricsUseCase by lazy { UpdateSystemMetricsUseCase(systemOverviewRepository) }
}