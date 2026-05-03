package com.gub.di

import com.gub.data.database.dao.SignalDao
import com.gub.data.database.dao.TrafficStatsDao
import com.gub.data.database.dao.WeatherDao
import com.gub.data.database.daoImpl.SignalDaoImpl
import com.gub.data.database.daoImpl.TrafficStatsDaoImpl
import com.gub.data.database.daoImpl.WeatherDaoImpl
import com.gub.data.repository.RepositoryAnalyticsImpl
import com.gub.data.repository.RepositoryDashboardImpl
import com.gub.data.repository.RepositoryMonitoringImpl
import com.gub.data.repository.RepositorySettingsImpl
import com.gub.data.service.common.WeatherService
import com.gub.data.service.dashboard.ServiceTrafficMeasure
import com.gub.data.service.dashboard.SystemOverviewService
import com.gub.data.service.monitoring.ServiceLiveSignal
import com.gub.domain.repository.RepositoryAnalytics
import com.gub.domain.repository.RepositoryDashboard
import com.gub.domain.repository.RepositoryMonitoring
import com.gub.domain.repository.RepositorySettings
import com.gub.domain.usecase.analytics.UseCaseTrafficVolume
import com.gub.domain.usecase.dashboard.UseCaseLiveTraffic
import com.gub.domain.usecase.dashboard.UseCaseSystemOverview
import com.gub.domain.usecase.monitoring.UseCaseLiveSignal
import com.gub.domain.usecase.settings.UseCaseExportCsv
import com.gub.domain.usecase.settings.UseCaseExportJson
import com.gub.domain.usecase.settings.UseCaseImportCsv
import com.gub.domain.usecase.settings.UseCaseImportJson
import org.koin.dsl.module
import sun.nio.ch.NativeThread.signal

val dashboardModule = module {
    // Data Access Objects (DAOs)
    single { SignalDaoImpl() as SignalDao }
    single { WeatherDaoImpl() as WeatherDao }
    single { TrafficStatsDaoImpl() as TrafficStatsDao }

    // Services
    single { ServiceLiveSignal() }
    single { WeatherService(get()) }
    single { ServiceTrafficMeasure(get(), get()) }
    single { SystemOverviewService(get(), get(), get()) }

    // Repository
    single { RepositoryAnalyticsImpl(get()) as RepositoryAnalytics }
    single { RepositoryMonitoringImpl(get()) as RepositoryMonitoring }
    single { RepositoryDashboardImpl(get(), get()) as RepositoryDashboard }
    single { RepositorySettingsImpl(get(), get(), get()) as RepositorySettings }

    // Use Cases
    single { UseCaseExportCsv(get()) }
    single { UseCaseExportJson(get()) }
    single { UseCaseImportCsv(get()) }
    single { UseCaseImportJson(get()) }

    single { UseCaseTrafficVolume(get()) }
    single { UseCaseLiveSignal(get()) }
    single { UseCaseLiveTraffic(get()) }
    single { UseCaseSystemOverview(get()) }
}