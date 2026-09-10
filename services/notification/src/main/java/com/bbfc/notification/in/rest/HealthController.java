package com.bbfc.notification.in.rest;

import java.time.Clock;
import java.util.HashMap;
import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
public class HealthController {
    private final Clock clock;

    public HealthController(Clock clock){
        this.clock = clock;
    }

    @GetMapping("/health")
    public Map<String, Object> health(){
        Map<String, Object> status = new HashMap<>();
        status.put("status", "UP");
        status.put("timestamp", clock.instant());
        return status;
    }
}
