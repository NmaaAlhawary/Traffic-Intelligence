package com.gub.domain.repository

interface RepositorySettings {

    suspend fun exportCsv(): String

    suspend fun exportJson(): String

    suspend fun importCsv(content: ByteArray)

    suspend fun importJson(content: ByteArray)
}