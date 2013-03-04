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
#include <assert.h>
#include <vigra/copyimage.hxx>
#include <vigra/numerictraits.hxx>

#include "slic_seeds.hxx"

namespace vigra{

// options for slic
struct SlicOptions{
    SlicOptions(
        const double m=10.0,
        const size_t iterations=40,
        const size_t sizeLimit=4
    ):
    m_(m),
    iterations_(iterations),
    sizeLimit_(sizeLimit){
    }
    const double m_;
    const size_t iterations_;
    const size_t sizeLimit_;
};



template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM=2>
class Slic{
public: 
    // 
    typedef DATA_IMAGE                          DataImageType;
    typedef LABEL_IMAGE                         LabelImageType; 
    // META-programing to get type information
    typedef typename detail_slic::GetValueType<DataImageType>::Type MaybeConstPixelType;
    typedef typename detail_slic::RemoveConstIfConst<MaybeConstPixelType>::Type PixelType;
    typedef typename detail_slic::GetValueType<LabelImageType>::Type MaybeConstLabelType;
    typedef typename detail_slic::RemoveConstIfConst<MaybeConstLabelType>::Type LabelType;
    typedef typename vigra::NumericTraits<PixelType>::RealPromote RealPromotePixelType;
    typedef typename vigra::NumericTraits<PixelType>::ValueType ValueType;
    typedef vigra::TinyVector<ValueType,2> SpatialCoordinateType;
    // Shape
    typedef vigra::TinyVector<int,2> ShapeType;
    //typedef vigra::TinyVector<int,2> ShapeType;
    // options type
    typedef SlicOptions OptionsType;

    // local classes which holds the color and spatial positon of centers (in vectors)
    struct CenterPosition{
        CenterPosition(){
            color_=0;
            spatial_=0;
        }
        RealPromotePixelType    color_;
        SpatialCoordinateType   spatial_;
    };

    Slic(DataImageType dataImage, const std::vector<SlicSeed<2> > & seeds, LabelImageType labelImage,const OptionsType & options=OptionsType()  );
    void initializeCenters();
    
    template<class COORDINATE_VALUE_TYPE>
    void getWindowLimits(const vigra::TinyVector<COORDINATE_VALUE_TYPE,2> &  centerCoord,vigra::TinyVector<int,2> & startCoord,vigra::TinyVector<int,2> & endCoord,const int radius)const;

    ValueType updateAssigments();
    void updateMeans();
    ValueType distance( const size_t  centerIndex,const vigra::TinyVector<int,2> &     pixelCoord)const;
    size_t postProcessing();


private:
    typedef vigra::MultiArray<2,ValueType>  DistanceImageType;

    DataImageType                   dataImage_;
    LabelImageType                  labelImage_;
    const std::vector<SlicSeed<2> > &   seeds_;
    const OptionsType               options_;
    vigra::MultiArray<2,LabelType>  tmpLabelImage_;
    DistanceImageType               distance_;
    const ShapeType                 shape_;
    std::vector<CenterPosition>     centers_;
    std::vector<size_t>             clusterSize_;
    const size_t                    k_;
    const ValueType                 mm_;
};



template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::Slic(

    Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::DataImageType                         dataImage, 
    const std::vector<SlicSeed<2> > &                                       seeds,
    Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::LabelImageType                        labelImage,
    const typename Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::OptionsType &          options 
)
:   dataImage_(dataImage),
    labelImage_(labelImage),
    seeds_(seeds),
    options_(options),
    tmpLabelImage_(ShapeType( dataImage.shape(0),dataImage.shape(1))),
    distance_( ShapeType( dataImage.shape(0),dataImage.shape(1)  )),
    shape_(dataImage.shape(0),dataImage.shape(1)),
    centers_(seeds.size()),
    clusterSize_(seeds.size()),
    k_(seeds.size()),
    mm_(static_cast<ValueType>(options_.m_*options_.m_))
{
    // initialize distance and labels
    std::fill(labelImage_.begin(),labelImage_.end(),static_cast<LabelType>(0));
    distance_=std::numeric_limits< ValueType >::infinity();
    // initialize the positon vector of the centers
    this->initializeCenters();

    // Do SLIC
    ValueType err=std::numeric_limits< ValueType >::infinity();
    for(size_t i=0;i<options_.iterations_;++i){
        // update which pixels gets assigned to which cluster
        const ValueType err2=this->updateAssigments();
        // update the mean
        this->updateMeans();//centers_,clusterSize_,k_);
        // convergence?
        if(err2+std::numeric_limits<ValueType>::epsilon()>=err){
            break;
        }
        err=err2;
    }
    // update assigments bevore postprocessing
    err=this->updateAssigments();
    size_t nBlocked=1;
    while(nBlocked!=0){
        //std::cout<<"nblocked = "<<nBlocked<<"\n";
        // remove all regions which are smaller than a sizeLimit
        nBlocked=this->postProcessing();
    }
    const size_t numLabels=vigra::labelImage(vigra::srcImageRange(labelImage_), vigra::destImage(tmpLabelImage_), false);
    tmpLabelImage_-=1;
    vigra::copyImage(vigra::srcImageRange(tmpLabelImage_), vigra::destImage(labelImage_));
}

template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
inline void 
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::initializeCenters(){
    for(size_t centerIndex=0;centerIndex<k_;++centerIndex){
        centers_[centerIndex].spatial_=seeds_[centerIndex].coordinates_;
        centers_[centerIndex].color_=dataImage_(seeds_[centerIndex].coordinates_[0],seeds_[centerIndex].coordinates_[1]);
    }
}

template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
typename Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::ValueType 
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::updateAssigments(){
    std::fill(distance_.begin(),distance_.end(),std::numeric_limits<ValueType>::infinity());
    std::fill(labelImage_.begin(),labelImage_.end(),0);
    for(size_t c=0;c<k_;++c){
        vigra::TinyVector<int,2> pixelCoord,startCoord,endCoord;
        // get window limits
        this->getWindowLimits(centers_[c].spatial_,startCoord,endCoord,seeds_[c].radius_);
        // => only pixels within the radius/searchSize of a cluster can be assigned to a cluster
        for(pixelCoord[1]=startCoord[1]; pixelCoord[1]<endCoord[1]; ++pixelCoord[1])
        for(pixelCoord[0]=startCoord[0]; pixelCoord[0]<endCoord[0]; ++pixelCoord[0]){
            // compute distance between cluster center and pixel
            ValueType dist=this->distance(c,pixelCoord);
            // update label?
            if(dist<=distance_(pixelCoord[0],pixelCoord[1])){
                labelImage_(pixelCoord[0],pixelCoord[1])=static_cast<LabelType>(c);
                distance_(pixelCoord[0],pixelCoord[1])=dist;
            }
        }
    }
    // compute error
    vigra::TinyVector<int,2> pixelCoord;
    ValueType totalDistance=static_cast<ValueType>(0);
    for(pixelCoord[1]=0 ;pixelCoord[1]<shape_[1];++pixelCoord[1])
    for(pixelCoord[0]=0 ;pixelCoord[0]<shape_[0];++pixelCoord[0]){
        const LabelType cluster=labelImage_(pixelCoord[0],pixelCoord[1]);
        ValueType dist=this->distance(cluster,pixelCoord);
        totalDistance+=dist;
    }
    return totalDistance;
}

template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
template<class COORDINATE_VALUE_TYPE>
inline void 
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::getWindowLimits
(
    const vigra::TinyVector<COORDINATE_VALUE_TYPE,2> &  centerCoord,
    vigra::TinyVector<int,2> &                          startCoord,
    vigra::TinyVector<int,2> &                          endCoord,
    const int radius
)const{
    for(size_t d=0;d<2;++d){
        startCoord[d] = std::max(int(0),int(vigra::round(float(centerCoord[d])))-int(radius));
        endCoord[d]   = std::min(shape_[d],int(vigra::round(float(centerCoord[d])))+radius+1);
    }
}


template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
void 
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::updateMeans(
){
    // get mean for each cluster
    // initialize cluster size with zeros 
    std::fill(clusterSize_.begin(),clusterSize_.begin()+k_,0);
    vigra::TinyVector<int,2> pixelCoord;
    for(pixelCoord[1]=0; pixelCoord[1]<shape_[1]; ++pixelCoord[1]){
        for(pixelCoord[0]=0; pixelCoord[0]<shape_[0]; ++pixelCoord[0]){
            const LabelType centerIndex=labelImage_(pixelCoord[0],pixelCoord[1]);
            if(clusterSize_[centerIndex]==0){
                centers_[centerIndex].color_  = dataImage_(pixelCoord[0],pixelCoord[1]);
                centers_[centerIndex].spatial_= pixelCoord;          
            }
            else{
                centers_[centerIndex].color_  += dataImage_(pixelCoord[0],pixelCoord[1]);
                centers_[centerIndex].spatial_+= pixelCoord;          
            }
            ++clusterSize_[centerIndex];
        }
    }
    // get mean 
    for(size_t centerIndex=0;centerIndex<k_;++centerIndex){
        const size_t size=clusterSize_[centerIndex];
        // if cluster has no members the old position stays
        if(size!=0){
            centers_[centerIndex].spatial_/=static_cast<ValueType>(size);
            centers_[centerIndex].color_ /=static_cast<ValueType>(size);
        }
    }
}

template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
inline typename Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::ValueType 
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::distance(
    const size_t                         centerIndex,
    const vigra::TinyVector<int,2> &     pixelCoord
)const{

    // SLIC costs
    const ValueType spatialDist   = vigra::squaredNorm(centers_[centerIndex].spatial_-pixelCoord);
    const ValueType colorDist     = vigra::squaredNorm(centers_[centerIndex].color_-dataImage_(pixelCoord[0],pixelCoord[1]));
    const ValueType normalization = static_cast<ValueType>(seeds_[centerIndex].radius_*seeds_[centerIndex].radius_);
    // FINAL COST
    return  colorDist + (spatialDist/normalization)*mm_;
}

template<class DATA_IMAGE ,class LABEL_IMAGE ,int DIM >
size_t 
Slic<DATA_IMAGE,LABEL_IMAGE,DIM>::postProcessing(){
    // get ride of disjoint regions:
    const size_t numLabels=vigra::labelImage(vigra::srcImageRange(labelImage_), vigra::destImage(tmpLabelImage_), false);
    tmpLabelImage_-=1;
    // copy the image which is free of disjoint regions back to the working image
    vigra::copyImage(vigra::srcImageRange(tmpLabelImage_), vigra::destImage(labelImage_));
    vigra::TinyVector<int,2> pixelCoord;
    std::vector< std::vector< vigra::TinyVector<int,2> > >  regionsPixels(numLabels);
    std::vector< std::vector<LabelType> >                   regionAdjacency(numLabels);
    // fill region adjacency graph
    for(pixelCoord[1]=0 ;pixelCoord[1]<shape_[1];++pixelCoord[1])
    for(pixelCoord[0]=0 ;pixelCoord[0]<shape_[0];++pixelCoord[0]){
        const LabelType l1=tmpLabelImage_(pixelCoord[0],pixelCoord[1]);
        if(pixelCoord[0]+1<shape_[0]){
            const LabelType l2=tmpLabelImage_(pixelCoord[0]+ 1 ,pixelCoord[1] );
            if(l1!=l2){
                regionAdjacency[l1].push_back(l2);
                regionAdjacency[l2].push_back(l1);
            }
        }
        if(pixelCoord[1]+1<shape_[1]){
            const LabelType l2=tmpLabelImage_(pixelCoord[0],pixelCoord[1]+1 );
            if(l1!=l2){
                regionAdjacency[l1].push_back(l2);
                regionAdjacency[l2].push_back(l1);
            }
        }        
        const LabelType label=tmpLabelImage_(pixelCoord[0],pixelCoord[1]);
        regionsPixels[label].push_back(pixelCoord);
    }
    // fill region size
    const size_t sizeLimit_=options_.sizeLimit_;
    std::vector<size_t> cSize(numLabels);
    for(size_t c=0;c<numLabels;++c){
        cSize[c]=regionsPixels[c].size();
    }
    size_t numChanges=0,blocked=0;
    std::vector<bool> merged(numLabels,false);
    //search for regions which are smaller than size limit
    for(size_t c=0;c<numLabels;++c){
        const size_t regionSize=regionsPixels[c].size();
        // a region is to small?
        if(regionSize<sizeLimit_ ){
            // the region is not a merged one (get ride of this check?)
            if(merged[c]==false){
                size_t mergeWith=0,maxSize=0;
                bool found=false; // search for a region to merge the smaller region with
                for(size_t i=0;i<regionAdjacency[c].size();++i){
                    const size_t c2=regionAdjacency[c][i];
                    assert(c2!=c);
                    const size_t size2=regionsPixels[c2].size();
                    if(size2 >=maxSize && merged[c2]==false){
                        found=true;
                        mergeWith=c2;
                        maxSize=size2;
                    }
                }
                // found a region to merge with?
                if(found){
                    cSize[mergeWith]+=regionSize;
                    // is there size > sizeLimit
                    if(cSize[mergeWith]<sizeLimit_){
                        merged[mergeWith]=true;
                        ++blocked;
                    }
                    merged[c]=true;
                    assert(c!=mergeWith);
                    ++numChanges;
                    for(size_t p=0;p<regionsPixels[c].size();++p){
                        ////std::cout<<"change stuff\n";
                        assert(labelImage_(regionsPixels[c][p][0],regionsPixels[c][p][1])!=mergeWith);
                        labelImage_(regionsPixels[c][p][0],regionsPixels[c][p][1])=mergeWith;
                    }
                }
                // did not found one.
                else{
                    ++blocked;
                }
            }
            // did not found one.
            else{
                ++blocked;
            }
        }
    }
    if(numChanges==0)
        return 0;
    return blocked;
}

} // end namespace vigra