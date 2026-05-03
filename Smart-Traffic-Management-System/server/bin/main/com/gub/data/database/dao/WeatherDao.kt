package com.gub.data.database.dao

import com.gub.data.database.entity.WeatherData

interface WeatherDao {

    fun insert(data: WeatherData)

    fun getAll(): List<WeatherData>

    fun getRecent(hours: Long): List<WeatherData>
}