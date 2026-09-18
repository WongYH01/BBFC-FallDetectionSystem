package com.bbfc.notification.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.List;

public interface AlertJPARepository extends JpaRepository<AlertEntity,String> {
    @Query(value = """
        SELECT * FROM alerts.alerts
         WHERE state IN ('DISPATCHED','ESCALATING')
           AND next_escalation_at <= :now
         ORDER BY next_escalation_at
           FOR UPDATE SKIP LOCKED
         LIMIT :limit
        """, nativeQuery = true)
    List<AlertEntity> findDueForEscalation(@Param("now") Instant now, @Param("limit") int limit);

}
