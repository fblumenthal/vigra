#ifndef VIGRA_SLIC_HELPER


template<int DIM>
struct Access;

template< >
struct Access<2>{

    template<class ARRAY,class COORDINTE_VECTOR>
    const typename ARRAY::value_type & const_ref(const ARRAY & array,const COORDINTE_VECTOR & coordinate){
        return array(coordinate[0],coordinate[1]);
    }
    template<class ARRAY,class COORDINTE_VECTOR>
    typename ARRAY::value_type & ref(ARRAY & array,const COORDINTE_VECTOR & coordinate){
        return array(coordinate[0],coordinate[1]);
    }
};


template< >
struct Access<3>{

    template<class ARRAY,class COORDINTE_VECTOR>
    const typename ARRAY::value_type & const_ref(const ARRAY & array,const COORDINTE_VECTOR & coordinate){
        return array(coordinate[0],coordinate[1],coordinate[2]);
    }
    template<class ARRAY,class COORDINTE_VECTOR>
    typename ARRAY::value_type & ref(ARRAY & array,const COORDINTE_VECTOR & coordinate){
        return array(coordinate[0],coordinate[1],coordinate[2]);
    }
};

#endif