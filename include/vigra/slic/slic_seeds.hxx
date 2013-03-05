#ifndef VIGRA_SLIC_SEEDS
#define VIGRA_SLIC_SEEDS

#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <numeric>
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/labelimage.hxx>
#include <vigra/imageiteratoradapter.hxx>
#include <iostream>
#include <map>
#include <assert.h>
#include <vigra/copyimage.hxx>
#include <vigra/numerictraits.hxx>


namespace vigra{
    namespace detail_slic{
        // metaprograming to remove the const of a type
        template<class T>
        struct RemoveConstIfConst{
            typedef T Type;
        };

        template<class T>
        struct RemoveConstIfConst<const T>{
            typedef T Type;
        };
        template<class T>
        struct GetValueType{
            typedef typename T::value_type Type;
        };

        template<class T>
        struct GetValueType<const T &>{
            typedef typename T::value_type Type;
        };

        template<class T>
        struct GetValueType<T &>{
            typedef typename T::value_type Type;
        };
    }

    // struct which holds a seed and the search radius / catchment area
    // of each seed (in this version it is not used that each seed could have a 
    // different catchment area)
    template<int DIMENSION>
    struct SlicSeed{
        SlicSeed():coordinates_(),radius_(){
        }
        SlicSeed(const vigra::TinyVector<int , DIMENSION> & coordinates, const int radius)
        :   coordinates_(coordinates),
            radius_(radius){
        }
        // This is need for vector index suite of boost python 
        bool operator == (const SlicSeed & other){ 
            for (int d=0;d<DIMENSION;++d)
                if(coordinates_[d] != other.coordinates_[d])   return false; 
            return true; 
        } 
    
        vigra::TinyVector<int , DIMENSION> coordinates_;
        int radius_;
    };

    typedef SlicSeed<2> SlicSeed2d;
    typedef SlicSeed<3> SlicSeed3d;
    typedef std::vector<SlicSeed<2> > SlicSeed2dVector;
    typedef std::vector<SlicSeed<3> > SlicSeed3dVector;


    // struct which holds the options for seed-generation
    // (k=number of clusters)
    // (r=radius in which one searchs for a minima of the gradient)
    struct  SlicSeedOptions{
        SlicSeedOptions(const size_t k,const size_t r=1) :
        k_(k),
        r_(r){
        }
        const size_t k_;
        const int r_;
    };


    // move slic seeds to a minima of the boundary indicator image (gradient-magnitude)
    template<class BOUNDARY_INDICATOR_IMAGE>
    void generateSlicSeedsImpl(
        BOUNDARY_INDICATOR_IMAGE boundaryIndicatorImage,
        std::vector<SlicSeed<2> >  &   seeds,
        const SlicSeedOptions  & options 
    ){
        typedef vigra::TinyVector<int , 2>                  CoordinateType;
        typedef BOUNDARY_INDICATOR_IMAGE                    BoundaryIndicatorImage;


        typedef typename detail_slic::GetValueType<BOUNDARY_INDICATOR_IMAGE>::Type MaybeConstValueType;
        typedef typename detail_slic::RemoveConstIfConst<MaybeConstValueType>::Type ValueType;

        seeds.clear();
        int shape[]={
            static_cast<int>(boundaryIndicatorImage.shape(0)),
            static_cast<int>(boundaryIndicatorImage.shape(1))
        };
        int seedDist=int(     std::sqrt(float(shape[0]*shape[1])/options.k_)  + 0.0f);
        //std::cout<<"seed dist "<<seedDist<<"\n";
        std::set<size_t> usedCenters;
        CoordinateType cCoord,startCoord,endCoord;

        for(cCoord[1] = seedDist/2; cCoord[1] < shape[1]; cCoord[1] += seedDist)
        for(cCoord[0] = seedDist/2; cCoord[0] < shape[0]; cCoord[0] += seedDist){
            // find min. gradient position in a window  of r
            //bool foundCenterPosition=false;
            ValueType minGrad=std::numeric_limits<ValueType>::infinity();
            // window limits
            for(size_t d=0;d<2;++d){
                startCoord[d] = std::max(cCoord[d]-options.r_, int(0) );
                endCoord[d]   = std::min(cCoord[d]+options.r_+1, shape[d] );
            }
            CoordinateType searchCoord,minCoord;
            for(searchCoord[1] = startCoord[1]; searchCoord[1]<endCoord[1]; ++searchCoord[1])
            for(searchCoord[0] = startCoord[0]; searchCoord[0]<endCoord[0]; ++searchCoord[0]){
                const ValueType grad=boundaryIndicatorImage(searchCoord[0],searchCoord[1]);
                // grad<minGrad
                if(grad<minGrad ){
                    const size_t key=searchCoord[0]+searchCoord[1]*shape[0];
                    // check if center position  is unused
                    if(usedCenters.find(key)==usedCenters.end()){
                        usedCenters.insert(key);
                        //foundCenterPosition=true;
                        minGrad=grad;
                        minCoord=searchCoord;
                    }
                }
            }
            // add seed
            seeds.push_back(SlicSeed<2>(minCoord,seedDist));
        }
    }

    template<class BOUNDARY_INDICATOR_IMAGE>
    void generateSlicSeedsImpl3d(
        BOUNDARY_INDICATOR_IMAGE boundaryIndicatorImage,
        std::vector<SlicSeed<3> >  &   seeds,
        const SlicSeedOptions  & options 
    ){
        typedef vigra::TinyVector<int , 3>                  CoordinateType;
        typedef BOUNDARY_INDICATOR_IMAGE                    BoundaryIndicatorImage;


        typedef typename detail_slic::GetValueType<BOUNDARY_INDICATOR_IMAGE>::Type MaybeConstValueType;
        typedef typename detail_slic::RemoveConstIfConst<MaybeConstValueType>::Type ValueType;

        seeds.clear();
        int shape[]={
            static_cast<int>(boundaryIndicatorImage.shape(0)),
            static_cast<int>(boundaryIndicatorImage.shape(1)),
            static_cast<int>(boundaryIndicatorImage.shape(2))
        };
        int seedDist=int(     std::sqrt(float(shape[0]*shape[1]*shape[2])/options.k_)  + 0.5f);
        //std::cout<<"seed dist "<<seedDist<<"\n";
        std::set<size_t> usedCenters;
        CoordinateType cCoord,startCoord,endCoord;

        for(cCoord[2] = 0; cCoord[2] < shape[2]; cCoord[2] += seedDist)
        for(cCoord[1] = 0; cCoord[1] < shape[1]; cCoord[1] += seedDist)
        for(cCoord[0] = 0; cCoord[0] < shape[0]; cCoord[0] += seedDist){
            // find min. gradient position in a window  of r
            //bool foundCenterPosition=false;
            ValueType minGrad=std::numeric_limits<ValueType>::infinity();
            // window limits
            for(size_t d=0;d<3;++d){
                startCoord[d] = std::max(cCoord[d]-options.r_, int(0) );
                endCoord[d]   = std::min(cCoord[d]+options.r_+1, shape[d] );
            }
            CoordinateType searchCoord,minCoord;

            for(searchCoord[2] = startCoord[2]; searchCoord[2]<endCoord[2]; ++searchCoord[2])
            for(searchCoord[1] = startCoord[1]; searchCoord[1]<endCoord[1]; ++searchCoord[1])
            for(searchCoord[0] = startCoord[0]; searchCoord[0]<endCoord[0]; ++searchCoord[0]){
                const ValueType grad=boundaryIndicatorImage(searchCoord[0],searchCoord[1],searchCoord[2]);
                // grad<minGrad
                if(grad<minGrad ){
                    const size_t key=searchCoord[0]+searchCoord[1]*shape[0]+searchCoord[2]*shape[0]*shape[1];
                    // check if center position  is unused
                    if(usedCenters.find(key)==usedCenters.end()){
                        usedCenters.insert(key);
                        //foundCenterPosition=true;
                        minGrad=grad;
                        minCoord=searchCoord;
                    }
                }
            }
            // add seed
            seeds.push_back(SlicSeed<3>(minCoord,seedDist));
        }
    }
}

#endif //VIGRA_SLIC_SEEDS
