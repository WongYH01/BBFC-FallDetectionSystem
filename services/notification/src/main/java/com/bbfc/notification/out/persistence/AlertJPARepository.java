package com.bbfc.notification.out.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

public interface AlertJPARepository extends JpaRepository<AlertEntity,String> {
    
}
