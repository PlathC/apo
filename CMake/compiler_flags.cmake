function(define_cxx_flags compile_flag_var compile_definitions_var link_options_var)
    # Reference: https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md
    if (MSVC)
        set(${compile_flag_var}
                ${${compile_flag_var}}
                /permissive
                /w14242
                /w14254
                /w14263
                /w14265
                /w14287
                /we4289
                /w14296
                /w14311
                /w14545
                /w14546
                /w14547
                /w14549
                /w14555
                /w14619
                /w14640
                /w14826
                /w14928
                /fp:fast
                PARENT_SCOPE)

        set(${compile_definitions_var} ${${compile_definitions_var}} NOMINMAX USE_MATH_DEFINES PARENT_SCOPE)
    else ()
        set(${compile_flag_var}
                ${${compile_flag_var}}
                -Wall
                -Wfatal-errors
                -Wextra
                -Wshadow
                -pedantic
                -Wnon-virtual-dtor
                -Wold-style-cast
                -Wcast-align
                -Wunused
                -Woverloaded-virtual
                -Wpedantic
                -Wconversion
                -Wsign-conversion
                -Wmisleading-indentation
                -Wduplicated-cond
                -Wduplicated-branches
                -Wlogical-op
                -Wnull-dereference
                -Wdouble-promotion
                -Wformat=2
                PARENT_SCOPE)
        set(${link_options_var} ${${link_options_var}} PARENT_SCOPE)
    endif ()
endfunction()

function(define_cuda_flags cuda_compile_flag_var)
    set(${cuda_compile_flag_var}
            ${${cuda_compile_flag_var}}
            --generate-line-info
            --use_fast_math
            --relocatable-device-code=true
            --extended-lambda

            # texture access deprecation...
            # __device__ annotation is ignored on a...
            -Xcudafe "--diag_suppress=1215 --diag_suppress=20012 --diag_suppress=20050"
            PARENT_SCOPE)
endfunction()
