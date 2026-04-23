package com.gub.features.settings.presentation.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.gub.features.settings.presentation.AlertNotificationsCard
import com.gub.features.settings.presentation.NotificationChannelsCard
import com.gub.features.settings.presentation.QuietHoursCard
import com.gub.features.settings.presentation.SystemNotificationsCard

@Composable
fun NotificationSettingsSection() {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            AlertNotificationsCard(modifier = Modifier.weight(1f))
            SystemNotificationsCard(modifier = Modifier.weight(1f))
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            NotificationChannelsCard(modifier = Modifier.weight(1f))
            QuietHoursCard(modifier = Modifier.weight(1f))
        }
    }
}