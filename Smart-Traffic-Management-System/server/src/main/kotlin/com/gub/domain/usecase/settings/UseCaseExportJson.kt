package com.gub.domain.usecase.settings

import com.gub.domain.repository.RepositorySettings

class UseCaseExportJson(
    private val repositorySettings: RepositorySettings
) {

    suspend operator fun invoke(): String {
        return repositorySettings.exportJson()
    }
}