package com.bbfc.notification.in.rest.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public record EventRequest (
    @NotBlank String eventId,
    @NotBlank String roomId,
    @NotBlank String roomName,
    @NotNull @DecimalMin("0.0") @DecimalMax("1.0") Double confidence
) {
 
}
