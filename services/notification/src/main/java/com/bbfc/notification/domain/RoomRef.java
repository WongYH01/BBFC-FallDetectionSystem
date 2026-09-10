package com.bbfc.notification.domain;

public record RoomRef(String roomId, String displayName) {
    public RoomRef{
        if(roomId == null || roomId.isBlank()){
            throw new IllegalArgumentException("roomId must not be empty");
        }
        if(displayName == null || displayName.isBlank()){
            throw new IllegalArgumentException("displayName must not be empty");
        }
    }
}
