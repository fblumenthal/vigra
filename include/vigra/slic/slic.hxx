// see slic_impl.hxx for implementation

#include <vector>
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/labelimage.hxx>
#include "slic_impl.hxx"


// generic
template<class BOUNDARY_INDICATOR_IMAGE>
inline void 
generateSlicSeeds(
    const BOUNDARY_INDICATOR_IMAGE      &  boundaryIndicatorImage,
    std::vector<SlicSeed>               &  seeds,
    const SlicSeedOptions               &  options 
){
    generateSlicSeedsImpl(boundaryIndicatorImage,seeds,options);
}


// BasicImage<RGBValue< ....> >
template< class DATA_IMAGE_TYPE,class GRADMAG_IMAGE_TYPE,class LABEL_IMAGE_TYPE>
inline void 
slicSuperpixels(
    DATA_IMAGE_TYPE          dataImage,
    GRADMAG_IMAGE_TYPE       gradMagImage,
    const std::vector<SlicSeed> &   seeds,
    LABEL_IMAGE_TYPE         labelImage,
    const SlicOptions &             parameter
){
    Slic< DATA_IMAGE_TYPE,GRADMAG_IMAGE_TYPE,LABEL_IMAGE_TYPE> slic( dataImage,gradMagImage,seeds,labelImage,parameter);
}



// BasicImage<RGBValue< ....> >
template< class DATA_IMAGE_TYPE,class GRADMAG_IMAGE_TYPE,class LABEL_IMAGE_TYPE>
inline void 
slicSuperpixelsMatlab(
    const DATA_IMAGE_TYPE    &      dataImage,
    const GRADMAG_IMAGE_TYPE   &    gradMagImage,
    const std::vector<SlicSeed> &   seeds,
    LABEL_IMAGE_TYPE        & labelImage,
    const SlicOptions &             parameter
){
    Slic< DATA_IMAGE_TYPE,GRADMAG_IMAGE_TYPE,LABEL_IMAGE_TYPE> slic( dataImage,gradMagImage,seeds,labelImage,parameter);
}