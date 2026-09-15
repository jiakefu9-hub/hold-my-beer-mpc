foreach(required CORE PREVIEW QUERY EXECUTE CMAKE_FILE ADAPTER_CMAKE)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

file(READ "${CORE}" core)
file(READ "${PREVIEW}" preview)
file(READ "${QUERY}" query)
file(READ "${EXECUTE}" execute)
file(READ "${CMAKE_FILE}" build_file)
file(READ "${ADAPTER_CMAKE}" adapter_build_file)

foreach(text IN ITEMS core preview query)
    foreach(forbidden IN ITEMS "ChannelPublisher" "LowCmd_" "rt/arm_sdk" "rt/lowcmd")
        if("${${text}}" MATCHES "${forbidden}")
            message(FATAL_ERROR "${text} contains forbidden command capability: ${forbidden}")
        endif()
    endforeach()
endforeach()

foreach(forbidden IN ITEMS "ReleaseMode" "SelectMode" "SetFsm" "SetVelocity"
                           "SetBalanceMode" "SetStandHeight")
    if(query MATCHES "${forbidden}")
        message(FATAL_ERROR "query source contains forbidden mutator: ${forbidden}")
    endif()
endforeach()

foreach(forbidden IN ITEMS "rt/lowcmd" "ReleaseMode" "SelectMode" "LocoClient"
                           "SetVelocity")
    if(execute MATCHES "${forbidden}")
        message(FATAL_ERROR "A2 execute source crosses scope: ${forbidden}")
    endif()
endforeach()

if(NOT execute MATCHES "rt/arm_sdk")
    message(FATAL_ERROR "A2 execute source lacks its sole allowed command topic")
endif()
if(NOT build_file MATCHES "G1_COMMISSIONING_BUILD_REAL_OUTPUT.*OFF")
    message(FATAL_ERROR "real output target must default OFF")
endif()
if(NOT build_file MATCHES "G1_COMMISSIONING_BUILD_DEVICE_QUERY.*OFF")
    message(FATAL_ERROR "device query target must default OFF")
endif()
if(NOT adapter_build_file MATCHES
       "Stage 2.*publisher|Stage 2.*DDS|真实hardware output尚未获授权")
    message(FATAL_ERROR "existing publisher-absent adapter guard disappeared")
endif()

message(STATUS "commissioning source/capability isolation verified")

if(DEFINED FSM_MONITOR)
    file(READ "${FSM_MONITOR}" fsm_monitor)
    foreach(forbidden IN ITEMS "ChannelPublisher" "LowCmd_" "rt/arm_sdk" "rt/lowcmd"
                               "ReleaseMode" "SelectMode" "SetFsm" "SetVelocity"
                               "ROBOT_API_ID_LOCO_SET_")
        if(fsm_monitor MATCHES "${forbidden}")
            message(FATAL_ERROR "FSM monitor contains command capability: ${forbidden}")
        endif()
    endforeach()
    foreach(required IN ITEMS "before_publisher" "before_write" "before_release"
                              "release_stop" "HandleArmStopState" "InterlockStop")
        if(NOT execute MATCHES "${required}")
            message(FATAL_ERROR "missing A2 interlock integration: ${required}")
        endif()
    endforeach()
endif()

if(DEFINED STOP_OBSERVER)
    file(READ "${STOP_OBSERVER}" stop_observer)
    foreach(forbidden IN ITEMS "ChannelPublisher" "LowCmd_" "rt/arm_sdk" "rt/lowcmd"
                               "ReleaseMode" "SelectMode" "SetFsm" "SetVelocity")
        if(stop_observer MATCHES "${forbidden}")
            message(FATAL_ERROR "stop observer contains command capability: ${forbidden}")
        endif()
    endforeach()
endif()

if(DEFINED MODE_STEP)
    file(READ "${MODE_STEP}" mode_step)
    foreach(forbidden IN ITEMS "ChannelPublisher" "LowCmd_" "rt/arm_sdk" "rt/lowcmd"
                               "ReleaseMode" "SelectMode" "SetVelocity"
                               "ROBOT_API_ID_LOCO_SET_BALANCE_MODE"
                               "ROBOT_API_ID_LOCO_SET_STAND_HEIGHT")
        if(mode_step MATCHES "${forbidden}")
            message(FATAL_ERROR "mode-step source crosses scope: ${forbidden}")
        endif()
    endforeach()
    if(NOT build_file MATCHES "G1_COMMISSIONING_BUILD_MODE_STEP.*OFF")
        message(FATAL_ERROR "mode-step target must default OFF")
    endif()
endif()
