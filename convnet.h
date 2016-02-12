#include <armadillo>

#include "checkerboard.h"

// DEBUG
#define PRINT_DEBUG {printf("\tFile: %s, Line: %i\n",__FILE__,__LINE__);}
using std::cerr;

class ConvNet
{


    public:
        void add_input( const BitBoard & bb );
};
