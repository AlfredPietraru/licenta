#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H
#include "Map.h"

class LocalMapping {
public:
    Map mapp;
    LocalMapping(Map mapp) {
        this->mapp = mapp;
    }

    void local_map(KeyFrame *kf);

};

#endif
