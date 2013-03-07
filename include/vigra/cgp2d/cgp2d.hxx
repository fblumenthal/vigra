#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <map>

#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>

#include "partition.hxx"

#include <vigra/multi_array.hxx>



#include <stdexcept>
#include <sstream>

#include "opengm/config.hxx"
#include "opengm/utilities/metaprogramming.hxx"

#define MYPI 3.14159265



#define CGP_ASSERT_OP(a,comp,b) \
    if(!  static_cast<bool>( a comp b )   ) { \
       std::stringstream s; \
       s << "OpenGM assertion " << #a <<#comp <<#b<< " failed:\n"; \
       s << #a "="<<a<<"\n"; \
       s << #b "="<<b<<"\n"; \
       s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
       throw std::runtime_error(s.str()); \
    }


    struct CellType{
        enum Values{
            Junction=0,
            Boundary=1,
            Region=2
        };
    };


namespace vigra {






    template<class LABEL_TYPE>
    class TopologicalGrid{
    public:
        typedef partition::Partition<size_t>    UdfType;
        typedef LABEL_TYPE LabelType;
        typedef vigra::MultiArray<2,LabelType>  LabelImageType;
        typedef typename LabelImageType::difference_type ShapeType;


        // constructor 
        template<class INPUT_IMG>
        TopologicalGrid(const INPUT_IMG & seg);

        // query
        const LabelImageType & tgrid()const;
        size_t numCells(const size_t i)const;
        size_t shape(const size_t d)const;
        LabelType operator()(const size_t tx,const size_t ty)     {return tgrid_(tx,ty);}
        LabelType operator()(const size_t tx,const size_t ty)const{return tgrid_(tx,ty);}
    private:
        size_t numCells_[3];
        LabelImageType tgrid_;
    };



    template<class COORDINATE_TYPE,class LABEL_TYPE>
    class Geometry;

    /*
    template<class COORDINATE_TYPE>
    struct BoundingBox{
        typedef Point<COORDINATE_TYPE>  PointType;
        BoundingBox(const PointType & upperLeft= PointType(),const PointType & lowerRight= PointType())
        :   upperLeft_(upperLeft),
            lowerRight_(lowerRight){
        }

        PointType upperLeft_;
        PointType lowerRight_;
    };


    template<class T>
    std::ostream& operator<<(std::ostream& os, const BoundingBox<T> & boundigBox) 
    { 
      os<<boundigBox.upperLeft_<<" / "<<boundigBox.lowerRight_;
      return os;
    } 
    */

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    class Cgp;

    template<class COORDINATE_TYPE,class LABEL_TYPE,int CELLTYPE>
    class CellBase{
        // friend classes
        friend class Cgp<COORDINATE_TYPE,LABEL_TYPE>;
    public:
        typedef COORDINATE_TYPE CoordinateType;
        typedef TinyVector<CoordinateType,2> PointType;
        //typedef BoundingBox<CoordinateType> BoundingBoxType;
        typedef LABEL_TYPE LabelType;
        size_t size()const{
            return points_.size();
        }
        const PointType & operator[](const size_t i) const{
            CGP_ASSERT_OP( i , < , points_.size() );
            return points_[i];
        }
        const PointType & operator[](const size_t i) {
            CGP_ASSERT_OP(i,<,points_.size());
            return points_[i];
        }
        LabelType label()const{
            return label_;
        }


        bool operator == (const CellBase & other){
            return label_==other.label_;
        }
        bool operator != (const CellBase & other){
            return label_!=other.label_;
        }
        /*
        BoundingBoxType boundingBox()const{
            PointType ul=points_[0];
            PointType lr=points_[0];
            for(size_t p=0;p<size();++p){
                ul.x_ = points_[p].x_ < ul.x_ ? points_[p].x_ : ul.x_;
                ul.y_ = points_[p].y_ < ul.y_ ? points_[p].y_ : ul.y_;
                lr.x_ = points_[p].x_ > lr.x_ ? points_[p].x_ : lr.x_;
                lr.y_ = points_[p].y_ > lr.y_ ? points_[p].y_ : lr.y_;
            }
            return BoundingBoxType(ul,lr);
        }
        

        template<class OUT_COORDINATE_TYPE>
        Point<OUT_COORDINATE_TYPE> centerCoordinate()const{
            OUT_COORDINATE_TYPE tx=static_cast<OUT_COORDINATE_TYPE>(0);
            OUT_COORDINATE_TYPE ty=static_cast<OUT_COORDINATE_TYPE>(0);
            for(size_t p=0;p<size();++p){
                tx+=static_cast<OUT_COORDINATE_TYPE>(points_[p].x_);
                ty+=static_cast<OUT_COORDINATE_TYPE>(points_[p].y_);
            }
            tx/=static_cast<OUT_COORDINATE_TYPE>(size());
            ty/=static_cast<OUT_COORDINATE_TYPE>(size());
            return Point<OUT_COORDINATE_TYPE>(tx,ty);
        }
        */
        const std::vector<LabelType> & bounds()const{
            return bounds_;
        }
        const std::vector<LabelType> & boundedBy()const{
            return boundedBy_;
        }

        const std::vector<PointType> & points()const{
            return points_;
        }
    protected:
        void push_back_point(const PointType & p){
            points_.push_back(p);
        }

        void push_back_bound(const LabelType l){
            bounds_.push_back(l);
        }

        void push_back_bounded_by(const LabelType l){
            boundedBy_.push_back(l);
        }

        void sortAdjaceny(){
            std::sort(bounds_.begin(),bounds_.end());
            std::sort(boundedBy_.begin(),boundedBy_.end());
        }

        void setLabel(const LabelType l){
            label_=l;
        }




        LabelType label_;
        // coordinates
        std::vector<PointType> points_;

        // bounds
        std::vector<LabelType> bounds_;
        std::vector<LabelType> boundedBy_;
        std::vector<LabelType> adjaceny_;
    };


    template<class COORDINATE_TYPE,class LABEL_TYPE,int CELLTYPE>
    class Cell;

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    class Cell<COORDINATE_TYPE,LABEL_TYPE,0> : public CellBase<COORDINATE_TYPE,LABEL_TYPE,0>{
    public:
        typedef LABEL_TYPE LabelType;
        typedef COORDINATE_TYPE CoordinateType;
        typedef TopologicalGrid<LabelType> TopologicalGridType;
        typedef TinyVector<CoordinateType,2> PointType;

        void getAngles(const TopologicalGridType & tgrid,const size_t radius=6){

            const int r=static_cast<int>(radius);
            const CoordinateType tx=this->points_[0][0],ty=this->points_[0][1];
            const CoordinateType xmin=    static_cast<int>(tx)-r < 0 ? 0 : static_cast<CoordinateType>(static_cast<int>(tx)-r );
            const CoordinateType ymin=    static_cast<int>(ty)-r < 0 ? 0 : static_cast<CoordinateType>(static_cast<int>(ty)-r );
            const CoordinateType xmax=    static_cast<int>(tx)+r+1 > tgrid.shape(0) ?   tgrid.shape(0) : static_cast<CoordinateType>(static_cast<int>(tx)+r+1 );
            const CoordinateType ymax=    static_cast<int>(ty)+r+1 > tgrid.shape(1) ?   tgrid.shape(1) : static_cast<CoordinateType>(static_cast<int>(ty)+r+1 );

            typedef std::pair<PointType,size_t>  MapItem;
            typedef std::map<LabelType,MapItem > AverageMapType;
            typedef typename AverageMapType::const_iterator AverageMapConstIter;
            typedef typename AverageMapType::iterator AverageMapIter;

            AverageMapType averageMap;
            for(size_t b=0;b<this->bounds_.size();++b){
                MapItem initItem= std::pair<PointType,size_t>(PointType(0,0),0);
                averageMap[b]=initItem;
            }

            // collect
            for(CoordinateType tyy=ymin;tyy<ymax;++tyy)
            for(CoordinateType txx=xmin;txx<xmax;++txx){
                // if boundary
                if(  (txx%2==1 && tyy%2==0) || (txx%2==0 && tyy%2==1) ){

                    LabelType cell1Label=tgrid(txx,tyy);
                    if(cell1Label!=0){
                        AverageMapIter iter=averageMap.find(cell1Label);
                        if(iter!=averageMap.end()){
                            MapItem item=iter->second;
                            item.first+=PointType(txx,tyy);
                            ++item.second;
                            averageMap[cell1Label]=item;
                        }
                    }
                }
            }
            // normalize 
            std::cout<<"degrees : ";
            for(AverageMapConstIter iter=averageMap.begin();iter!=averageMap.end();++iter){\
                MapItem item=iter->second;
                PointType averagePoint = item.first;
                averagePoint/=item.second;

                const float x=static_cast<float>(tx);
                const float y=static_cast<float>(ty);
                const float ax=static_cast<float>(averagePoint[0]);
                const float ay=static_cast<float>(averagePoint[1]);

                const float rx=ax-x;
                const float ry=ay-y;

                const float result = std::atan2 (ry,rx) * 180.0 / MYPI;

                std::cout<<" "<<result;

            }
            std::cout<<"\n";
        }
    };

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    class Cell<COORDINATE_TYPE,LABEL_TYPE,1> : public CellBase<COORDINATE_TYPE,LABEL_TYPE,1>{
    public:
        typedef LABEL_TYPE LabelType;
        typedef COORDINATE_TYPE CoordinateType;
        typedef TopologicalGrid<LabelType> TopologicalGridType;
        typedef TinyVector<CoordinateType,2> PointType;
    private:
    };

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    class Cell<COORDINATE_TYPE,LABEL_TYPE,2> : public CellBase<COORDINATE_TYPE,LABEL_TYPE,2>{
    public:
        typedef LABEL_TYPE LabelType;
        typedef COORDINATE_TYPE CoordinateType;
        typedef TopologicalGrid<LabelType> TopologicalGridType;
        typedef TinyVector<CoordinateType,2> PointType;
    private:
    };

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    class Cgp{
    public:
        typedef LABEL_TYPE LabelType;
        typedef COORDINATE_TYPE CoordinateType;
        typedef TopologicalGrid<LabelType> TopologicalGridType;
        //typedef BoundingBox<CoordinateType> BoundingBoxType;
        typedef TinyVector<CoordinateType,2> PointType;
        typedef Cell<COORDINATE_TYPE,LABEL_TYPE,0> GeoCell0;
        typedef Cell<COORDINATE_TYPE,LABEL_TYPE,1> GeoCell1;
        typedef Cell<COORDINATE_TYPE,LABEL_TYPE,2> GeoCell2;
        typedef std::vector< GeoCell0 > GeoCells0;
        typedef std::vector< GeoCell1 > GeoCells1;
        typedef std::vector< GeoCell2 > GeoCells2;

        // Constructor
        Cgp(const TopologicalGridType & tgrid );
        // Query
        const GeoCells0  & geometry0()const;
        const GeoCells1  & geometry1()const;
        const GeoCells2  & geometry2()const;

    private:
        std::vector< GeoCell0 >  geoCells0_;
        std::vector< GeoCell1 >  geoCells1_;
        std::vector< GeoCell2 >  geoCells2_;
    };


    // Implementation ToopogicalGrid
    template<class LABEL_TYPE>
    template<class INPUT_IMG>
    TopologicalGrid<LABEL_TYPE>::TopologicalGrid(const INPUT_IMG & seg)
    : tgrid_(ShapeType(seg.shape(0)*2-1,seg.shape(1)*2-1))
    {
        const size_t dx=seg.shape(0);
        const size_t dy=seg.shape(1);
        const size_t tdx=seg.shape(0)*2-1;
        const size_t tdy=seg.shape(1)*2-1;
        size_t shape[] = { tdx,tdy};
        // counters
        size_t maxRegionLabel=0; //TODO TODO TODO
        size_t junctionIndex=0;
        size_t boundaryElementLabel=1;
        ////////////////
        // 1. PASS //
        ////////////////
        for(size_t ty=0;ty<tdy;++ty)
        for(size_t tx=0;tx<tdx;++tx){
            //std::cout<<" tx "<<tx<<" ty "<<ty<<"\n";
            // if region
            if(tx%2==0 && ty%2==0){
                size_t label=seg(tx/2,ty/2);
                tgrid_(tx,ty)=label;
                maxRegionLabel=label>maxRegionLabel ? label : maxRegionLabel;
            }
            // if junction
            else if(tx%2!=0 && ty%2!=0){
                //  A|B
                //  _ _
                //  C|D
                // check labels of A,B,C and D
                std::set<LABEL_TYPE> lset;
                lset.insert( seg((tx-1)/2,(ty-1)/2));  // A
                lset.insert( seg((tx+1)/2,(ty-1)/2));  // B
                lset.insert( seg((tx-1)/2,(ty+1)/2));  // A
                lset.insert( seg((tx+1)/2,(ty+1)/2));  // A
                if(lset.size()>=3){
                    tgrid_(tx,ty)=junctionIndex+1;
                    ++junctionIndex;
                }
                else{
                    tgrid_(tx,ty)=0;
                }
            }
            // boundary
            else{
                size_t l0,l1;
                // A|B
                // vertical  boundary 
                if(tx%2==1){
                    l0=seg( (tx-1)/2, ty/2 );
                    l1=seg( (tx+1)/2, ty/2 );
                }
                // horizontal boundary
                else{
                    l0=seg( tx/2, (ty-1)/2);
                    l1=seg( tx/2, (ty+1)/2);
                }
                // active boundary ?
                if(l0!=l1){
                    //std::cout<<l0<<"/"<<l1<<"\n";
                    tgrid_(tx,ty)=boundaryElementLabel;
                    ++boundaryElementLabel;
                }
                else
                    tgrid_(tx,ty)=0;
            }
        }
        /////////////////
        // 2. PASS //
        /////////////////
        UdfType boundaryUdf(boundaryElementLabel-1);
        const size_t num1Elements=boundaryElementLabel-1;
        for(size_t ty=0;ty<tdy;++ty)
        for(size_t tx=0;tx<tdx;++tx){
            // boundary
            if((tx%2!=0 && ty%2!=1 ) || (tx%2!=1 && ty%2!=0 )) {
                if ( tgrid_(tx,ty)!=0){
                    size_t ownIndex=tgrid_(tx,ty);
                    // vertical boundary
                    if(tx%2==1){
                        // each horizontal boundary has 6 candidate neighbours:
                        //  _|_
                        //  _|_  <= this is the boundary we are looking right now
                        //   |
                        // if junction down is inctive
                        LabelType other;
                        if (ty+1 < tdy  && tgrid_(tx,ty+1)==0){
                            // boundary up is active?
                            other=tgrid_(tx,ty+2);
                            if( other!=0){
                                CGP_ASSERT_OP(other-1, <, num1Elements);
                                boundaryUdf.merge(ownIndex-1,other-1);
                            }
                            // boundary left is active?
                            other=tgrid_(tx-1,ty+1);
                            if( other!=0){
                                CGP_ASSERT_OP(other-1 ,<, num1Elements);
                                boundaryUdf.merge(ownIndex-1,other-1 );
                            }
                            // boundary right is active?
                            if( tgrid_(tx+1,ty+1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx+1,ty+1)-1 );
                            }
                        }
                        // if junction up is inctive
                        if(ty > 0 && tgrid_(tx,ty-1)==0){
                            // boundary up is active?
                            if( tgrid_(tx,ty-2)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx,ty-2) -1);
                            }
                            // boundary left is active?
                            if( tgrid_(tx-1,ty-1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx-1,ty-1)-1 );
                            }
                            // boundary right is active?
                            if( tgrid_(tx+1,ty-1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx+1,ty-1)-1 );
                            }
                        }
                    }
                    // horizontal boundary 
                    else{
                        //   each horizontal boundary has 6 candidate neighbours:
                        //   _|_|_
                        //    | |
                        // 
                        // if left junction inactive?     
                        if(tx >0 && tgrid_( tx-1,ty)==0){
                            // boundary left is active?
                            if( tgrid_(tx-2,ty)!=0){
                                //std::cout<<"merge left \n";
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx-2,ty)-1 );
                            }
                            // boundary up is active?
                            if( tgrid_(tx-1,ty-1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx-1,ty-1)-1 );
                            }
                            // boundary down is active?
                            if( tgrid_(tx-1,ty+1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx-1,ty+1)-1 );
                            }
                        }
                        // if right junction inactive?     
                        if(tx+1<tdx &&tgrid_( tx+1,ty)==0){
                            // boundary right is active?
                            if( tgrid_(tx+2,ty)!=0){
                                //std::cout<<"merge right \n";
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx+2,ty)-1 );
                            }
                            // boundary up is active?
                            if( tgrid_(tx+1,ty-1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx+1,ty-1)-1 );
                            }
                            // boundary down is active?
                            if( tgrid_(tx+1,ty+1)!=0){
                                boundaryUdf.merge(ownIndex-1,tgrid_(tx+1,ty+1)-1 );
                            }
                        }
                    }
                }
            }
        }

        // dense relabeling
        std::map<size_t,size_t> relabel;
        boundaryUdf.representativeLabeling(relabel);
        /////////////////
        // 3. PASS //
        /////////////////
        for(size_t ty=0;ty<tdy;++ty)
        for(size_t tx=0;tx<tdx;++tx){
            // boundary
            if((tx%2!=0 && ty%2!=1 ) || (tx%2!=1 && ty%2!=0 )) {
                if(tgrid_(tx,ty)!=0){
                    // relabel
                    size_t notDenseIndex=boundaryUdf.find( tgrid_(tx,ty)-1 );
                    size_t denseIndex=relabel[notDenseIndex];
                    tgrid_(tx,ty)=denseIndex+1;
                }
            }
        }

        // update cell counters
        numCells_[CellType::Region]=maxRegionLabel;
        CGP_ASSERT_OP(boundaryUdf.numberOfSets(),==,relabel.size());
        numCells_[CellType::Boundary]=relabel.size();
        numCells_[CellType::Junction]=junctionIndex;
    }

    template<class LABEL_TYPE>
    const typename TopologicalGrid<LABEL_TYPE>::LabelImageType & TopologicalGrid<LABEL_TYPE>::tgrid()const{
        return tgrid_;
    }

    template<class LABEL_TYPE>
    size_t TopologicalGrid<LABEL_TYPE>::numCells(const size_t i)const{
        return numCells_[i];
    }

    template<class LABEL_TYPE>
    size_t TopologicalGrid<LABEL_TYPE>::shape(const size_t d)const{
        return tgrid_.shape(d);
    }

    // Implementation Cgp
    template<class COORDINATE_TYPE,class LABEL_TYPE>
    Cgp<COORDINATE_TYPE,LABEL_TYPE>::Cgp(const typename Cgp<COORDINATE_TYPE,LABEL_TYPE>::TopologicalGridType  & tgrid )
    :   geoCells0_(tgrid.numCells(0)),
        geoCells1_(tgrid.numCells(1)),
        geoCells2_(tgrid.numCells(2))
    {
        // set up geometry
        const typename TopologicalGridType::LabelImageType & grid=tgrid.tgrid();
        for(size_t ty=0;ty<tgrid.shape(1);++ty)
        for(size_t tx=0;tx<tgrid.shape(0);++tx){
            // Cell 2 (=Region)
            if(tx%2==0 && ty%2==0){
                int label = grid(tx,ty);
                CGP_ASSERT_OP(label,>,0)
                CGP_ASSERT_OP(label,<=,tgrid.numCells(2));
                geoCells2_[label-1].push_back_point(PointType(tx,ty));
            }
            // Cell 0 (== Junction)
            else if(tx%2!=0 && ty%2!=0){
                int label = grid(tx,ty);
                if(label!=0){
                    CGP_ASSERT_OP(label,>,0)
                    CGP_ASSERT_OP(label,<=,tgrid.numCells(0));
                    geoCells0_[label-1].push_back_point(PointType(tx,ty));
                }
                    
            }
            // Cell 1 (== Boundary)
            else{
                int label = grid(tx,ty);
                if(label!=0){
                    CGP_ASSERT_OP(label,>,0)
                    CGP_ASSERT_OP(label,<=,tgrid.numCells(1));
                    geoCells1_[label-1].push_back_point(PointType(tx,ty));
                }
            }
        }
        // check size of geometry
        CGP_ASSERT_OP(geoCells0_.size(),==,tgrid.numCells(0));
        CGP_ASSERT_OP(geoCells1_.size(),==,tgrid.numCells(1));
        CGP_ASSERT_OP(geoCells2_.size(),==,tgrid.numCells(2));
        // set up bounds and bounded by

        // iterate over all 0-cells / junctions
        for(size_t cell0Index=0;cell0Index<tgrid.numCells(0);++cell0Index){
            const LabelType cell0Label=cell0Index+1;
            // set up label
            geoCells0_[cell0Index].setLabel(cell0Label);
            // get coordinates
            const size_t tx=geoCells0_[cell0Index][0][0];
            const size_t ty=geoCells0_[cell0Index][0][1];
            // Loop over all possible Cell1's / boundaries of the Cell0 / Junction
            const int px[]={ 1, -1, 0, 0};
            const int py[]={ 0,  0, 1,-1};
            for(size_t b=0;b<4;++b){
                LabelType cell1Label=grid(int(tx)+px[b],int(ty)+py[b]);
                // check if Cell1 / boundary is active
                if(cell1Label!=0){

                    CGP_ASSERT_OP(cell1Label,>,0)
                    CGP_ASSERT_OP(cell1Label,<=,tgrid.numCells(1));

                    LabelType cell1Index=cell1Label-1;
                    // bounds ( boundaries of a juction)
                    geoCells0_[cell0Index].push_back_bound(cell1Label);
                    // junctions of a boundaty
                    geoCells1_[cell1Index].push_back_bounded_by(cell0Label);
                }
            }
            CGP_ASSERT_OP(geoCells0_[cell0Index].bounds().size(),>=,3);
            CGP_ASSERT_OP(geoCells0_[cell0Index].bounds().size(),<=,4);
        }

        // iterate over all 1-cells / boundaries
        for(size_t cell1Index=0;cell1Index<tgrid.numCells(1);++cell1Index){
            const LabelType cell1Label=cell1Index+1;
            // set up label
            geoCells1_[cell1Index].setLabel(cell1Label);
            // get tx and ty of SOME element of the boundary (the first in this case) 
            const size_t tx=geoCells1_[cell1Index][0][0];
            const size_t ty=geoCells1_[cell1Index][0][1];
            // bounds (region labels)
            LabelType cell2LabelA,cell2LabelB;
            // vertical boundary
            if(tx%2==1){
                cell2LabelA=static_cast<LabelType>(grid(tx-1,ty));
                cell2LabelB=static_cast<LabelType>(grid(tx+1,ty));

            }
            else{
                cell2LabelA=static_cast<LabelType>(grid(tx,ty-1));
                cell2LabelB=static_cast<LabelType>(grid(tx,ty+1));
            }
            CGP_ASSERT_OP(cell2LabelA,>,0)
            CGP_ASSERT_OP(cell2LabelA,<=,tgrid.numCells(2));
            CGP_ASSERT_OP(cell2LabelB,>,0)
            CGP_ASSERT_OP(cell2LabelB,<=,tgrid.numCells(2));
            const LabelType cell2IndexA=cell2LabelA-1;
            const LabelType cell2IndexB=cell2LabelB-1;

            // set up bounds (the 2 adj. regions to this boundary)
            geoCells1_[cell1Index].push_back_bound(cell2LabelA);
            geoCells1_[cell1Index].push_back_bound(cell2LabelB);
            // set up bounded by ( n adj. boundaries of a region)
            geoCells2_[cell2IndexA].push_back_bounded_by(cell1Label);
            geoCells2_[cell2IndexB].push_back_bounded_by(cell1Label);
        }
        // sortAdjaceny

        // iterate over all 2-cells / regions 
        for(size_t cell2Index=0;cell2Index<tgrid.numCells(2);++cell2Index){
            // set up label
            geoCells2_[cell2Index].setLabel(cell2Index+1);
            // sortAdjaceny
            geoCells2_[cell2Index].sortAdjaceny();
        }
        // iterate over all 1-cells / boundaries
        for(size_t cell1Index=0;cell1Index<tgrid.numCells(1);++cell1Index){
            // sortAdjaceny
            geoCells1_[cell1Index].sortAdjaceny();
        }
        // iterate over all 0-cells / junctions
        for(size_t cell0Index=0;cell0Index<tgrid.numCells(0);++cell0Index){
            // sortAdjaceny
            geoCells0_[cell0Index].setLabel(cell0Index);
            // sortAdjaceny
            geoCells0_[cell0Index].sortAdjaceny();
        }

    }

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    const typename Cgp<COORDINATE_TYPE,LABEL_TYPE>::GeoCells0  & 
    Cgp<COORDINATE_TYPE,LABEL_TYPE>::geometry0()const{
        return geoCells0_;
    }

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    const typename Cgp<COORDINATE_TYPE,LABEL_TYPE>::GeoCells1  & 
    Cgp<COORDINATE_TYPE,LABEL_TYPE>::geometry1()const{
        return geoCells1_;
    }

    template<class COORDINATE_TYPE,class LABEL_TYPE>
    const typename Cgp<COORDINATE_TYPE,LABEL_TYPE>::GeoCells2  & 
    Cgp<COORDINATE_TYPE,LABEL_TYPE>::geometry2()const{
        return geoCells2_;
    }

}