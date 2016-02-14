
#include "checkerboard.h"
#include <armadillo>

// Feed Forward Neural Net
class ffnn
{
    public:
        void set_board( BitBoard );
        void feed_forward();
        double get_output();

};
