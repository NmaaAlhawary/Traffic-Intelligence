package com.gub.domain.usecase.settings

import com.gub.domain.repository.RepositorySettings

class UseCaseExportCsv(
    private val repositorySettings: RepositorySettings
) {

    suspend operator fun invoke(): String {
        return repositorySettings.exportCsv()
    }
}