#ifndef VIGRA_SLIC
#define VIGRA_SLIC

#include <vector>
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/labelimage.hxx>
#include "slic/slic_impl.hxx"
#include "slic/slic_seeds.hxx"


namespace vigra{

    template<class BOUNDARY_INDICATOR_IMAGE>
    inline void 
    generateSlicSeeds(
        const BOUNDARY_INDICATOR_IMAGE      &  boundaryIndicatorImage,
        std::vector<SlicSeed>               &  seeds,
        const SlicSeedOptions               &  options 
    ){
        generateSlicSeedsImpl <const BOUNDARY_INDICATOR_IMAGE & >( boundaryIndicatorImage,seeds,options);
    }


    template< class DATA_IMAGE_TYPE,class LABEL_IMAGE_TYPE>
    inline void 
    slicSuperpixels(
        const DATA_IMAGE_TYPE &         dataImage,
        const std::vector<SlicSeed> &   seeds,
        LABEL_IMAGE_TYPE  &             labelImage,
        const SlicOptions &             parameter
    ){
        Slic< const DATA_IMAGE_TYPE & ,LABEL_IMAGE_TYPE & > slic( dataImage,seeds,labelImage,parameter);
    }

}

#endif // VIGRA_SLIC