package com.gub.features.monitoring.di

import com.gub.di.NetworkModule.provideHttpClient
import com.gub.features.monitoring.data.network.LiveSignalApi
import com.gub.features.monitoring.data.network.LiveSignalApiImpl
import com.gub.features.monitoring.data.repository.RepositoryLiveSignalImpl
import com.gub.features.monitoring.domain.repository.RepositoryLiveSignal
import com.gub.features.monitoring.domain.usecase.UseCaseLiveSignal

object MonitoringModule {

    private val liveSignalApi: LiveSignalApi by lazy {
        LiveSignalApiImpl(provideHttpClient())
    }

    private val repositoryLiveSignal: RepositoryLiveSignal by lazy {
        RepositoryLiveSignalImpl(liveSignalApi)
    }

    val getUseCaseLiveSignal by lazy { UseCaseLiveSignal(repositoryLiveSignal) }
}