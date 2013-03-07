#include "cgp2d.hxx"



int main(){

    std::size_t data[]={
        0,0,2,0,
        0,0,2,0,
        1,1,1,1,
        1,1,1,1
    };
    std::size_t shape[]={4,4};
    marray::Marray<std::size_t> seg(shape,shape+2);
    for(size_t i=0;i<seg.size();++i){
        seg(i)=data[i];
    }

    TopologicalGrid tgrid(seg);

    for(size_t y=0;y<shape[1]*2-1;++y){
        for(size_t x=0;x<shape[0]*2-1;++x){
            if(tgrid.tgrid()(x,y)==-1){
                std::cout<<" -";
            }
            else{
                std::cout<<" "<<tgrid.tgrid()(x,y);
            }
        }
        std::cout<<"\n";
    }

    std::cout<<"num Regions    "<<tgrid.numCells(CellType::Region)<<"\n";
    std::cout<<"num Boundaries "<<tgrid.numCells(CellType::Boundary)<<"\n";
    std::cout<<"num Junctions  "<<tgrid.numCells(CellType::Junction)<<"\n";

    typedef Geometry<short,int> GeometryType;
    typedef GeometryType::BoundingBoxType BoundingBoxType;
    GeometryType geo(tgrid);

    typedef GeometryType::GeoCells0 GeoCells0;
    typedef GeometryType::GeoCells1 GeoCells1;
    typedef GeometryType::GeoCells2 GeoCells2;

    const GeoCells0 & geo0=geo.geometry0();
    const GeoCells1 & geo1=geo.geometry1();
    const GeoCells2 & geo2=geo.geometry2();

    std::cout<<"0Cells:\n\n";
    for(size_t i=0;i<tgrid.numCells(0);++i){
        size_t geoSize=geo0[i].size();
        std::cout<<"boundingBox : "<<geo0[i].boundingBox()<<"\n";
        std::cout<<"center : "<<geo0[i].centerCoordinate<float>()<<"\n";
        std::cout<<" label "<<i<<"\n";
        std::cout<<"    size "<<geoSize<<"\n";
    }
    std::cout<<"1Cells:\n\n";
    for(size_t i=0;i<tgrid.numCells(1);++i){
        size_t geoSize=geo1[i].size();
        std::cout<<"boundingBox : "<<geo1[i].boundingBox()<<"\n";
        std::cout<<"center : "<<geo1[i].centerCoordinate<float>()<<"\n";
        std::cout<<" label "<<i<<"\n";
        std::cout<<"    size "<<geoSize<<"\n";
    }
    std::cout<<"2Cells:\n\n";
    for(size_t i=0;i<tgrid.numCells(2);++i){
        size_t geoSize=geo2[i].size();
        std::cout<<"boundingBox : "<<geo2[i].boundingBox()<<"\n";
        std::cout<<"center : "<<geo2[i].centerCoordinate<float>()<<"\n";
        std::cout<<" label "<<i<<"\n";
        std::cout<<"    size "<<geoSize<<"\n";
    }
}