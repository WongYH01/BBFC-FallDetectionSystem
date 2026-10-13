package com.bbfc.notification.core.domain;

public record Confidence(double confidenceScore) {
    public Confidence{
        if(confidenceScore < 0.0 || confidenceScore > 1.0){
            throw new IllegalArgumentException("Confidence must be between 0 and 1, received " + confidenceScore);
        }
    }

    public enum Band {
        LOW,
        MEDIUM,
        HIGH
    }

    public Band band() {
        if (confidenceScore>=0.8) return Band.HIGH; 
        if (confidenceScore>=0.5) return Band.MEDIUM; 
        return Band.LOW; 
    }
}
