package com.gub.domain.usecase.dashboard

import com.gub.domain.models.dashboard.ModelAiControl
import com.gub.domain.repository.RepositoryDashboard

class UseCaseAiControlSystem(private val repositoryDashboard: RepositoryDashboard) {

    operator fun invoke(): ModelAiControl {
        return repositoryDashboard.getAiControlSystem()
    }
}