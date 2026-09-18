package com.bbfc.notification.config;

import com.bbfc.notification.core.domain.EscalationPolicy;
import com.bbfc.notification.core.domain.FixedEscalationPolicy;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class EscalationPolicyConfig {
    @Bean
    public EscalationPolicy escalationPolicy(EscalationConfig escalationConfig){
        return new FixedEscalationPolicy(escalationConfig.window(), escalationConfig.maxRepeats());
    }
}
