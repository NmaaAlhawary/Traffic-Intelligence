package com.gub.data.models

import com.gub.data.database.entity.SignalData
import com.gub.data.database.entity.TrafficStats
import com.gub.data.database.entity.WeatherData

data class JsonExportWrapper(
    val trafficStats: List<TrafficStats>,
    val signalData: List<SignalData>,
    val weatherData: List<WeatherData>
)