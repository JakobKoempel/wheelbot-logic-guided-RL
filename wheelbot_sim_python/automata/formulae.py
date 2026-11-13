
turn_left = '(d&!rv&!(lr|a|s)) U (' \
            '(d&(lr|a|s)) & X (' \
                '(d&lr&!(a|s)) U (' \
                    '(d&(a|s)) & X (' \
                        '(d&a&!s) U (d&s)' \
                    ')' \
                ')' \
            ')' \
        ')'

turn_right = '(d&rv&!(rr|a|s)) U (' \
            '(d&(rr|a|s)) & X (' \
                '(d&rr&!(a|s)) U (' \
                    '(d&(a|s)) & X (' \
                        '(d&a&!s) U (d&s)' \
                    ')' \
                ')' \
            ')' \
        ')'

turn_either = f'(rv => X ({turn_right})) & (!rv => X ({turn_left}))'

circle_left = '(d & !rv & !lr) U G (lr & d)'

circle_right = '(d & rv & !rr) U G (rr & d)'

align_left = '(d&!rv&!(lr|a)) U (' \
            '(d&(lr|a)) & X (' \
                '(d&lr&!a) U (' \
                    '(d&a)' \
                ')' \
            ')' \
        ')'

align_right = '(d&rv&!(rr|a)) U (' \
            '(d&(rr|a)) & X (' \
                '(d&rr&!a) U (' \
                    '(d&a)' \
                ')' \
            ')' \
        ')'