package com.gub.data.database.dao

import com.gub.data.database.entity.SignalData

interface SignalDao {

    fun insert(data: SignalData)

    fun getByRouteId(id: String): List<SignalData>

    fun getAll(): List<SignalData>

    fun getByHour(hour: Int): List<SignalData>

    fun delete(id: String)
}