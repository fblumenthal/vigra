#define CLASSIFIER_TEST 1
#define HDF5 0
#define CROSSVAL 0
#ifdef _MSC_VER
# pragma warning (disable : 4244)
#endif
#include <iostream>
#include <fstream>
#include <functional>
#include <cmath>
#include <vigra/random_forest.hxx>
#include <vigra/random_forest/rf_more_splits.hxx>
#include <vigra/random_forest_deprec.hxx>
#include <vector>
#include <limits>
//#include "data/RF_results.hxx"
#include "data/RF_data.hxx"
#include "test_visitors.hxx"
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vigra/stdimage.hxx>
#include <vigra/unittest.hxx>
#include "vigra/hdf5impex.hxx"
#include "vigra/multi_array.hxx"
#include "vigra/random_forest_hdf5_impex.hxx"
//We need to undefine NDEBUG so that we have TIC, TOC available!
#undef NDEBUG
#include <vigra/timing.hxx>
#include <vigra/impex.hxx>

/// Include here the files for the random regresion forest

USETICTOC
;

using namespace vigra;

struct RegressionForestTest
{
    /**
     * 1D Regression test. Checks that the results does not differ from
     * a reference which was manually checked. The paremeters of the rf should be fixed.
     */
    void RegressionForestTest1D()
    {


        std::cout << "Entering the Regression Forest test \n";
        std::string filename="data/regression1D_noise.h5";

        std::string path1="features_train";
        HDF5ImportInfo infoFeatures(filename.c_str(),path1.c_str() );
        MultiArrayShape<2>::type shpFeatures(infoFeatures.shape()[0],infoFeatures.shape()[1]);
        MultiArray<2, double> features(shpFeatures);
        readHDF5(infoFeatures, features);

        std::string path2="labels_train";
        HDF5ImportInfo infoLabels(filename.c_str(),path2.c_str() );
        MultiArrayShape<2>::type shpLabels(infoLabels.shape()[0],infoLabels.shape()[1]);
        MultiArray<2, double> labels(shpLabels);
        readHDF5(infoLabels, labels);

        std::string path3="features_test";
        HDF5ImportInfo infoFeatures2(filename.c_str(),path3.c_str() );
        MultiArrayShape<2>::type shpFeatures2(infoFeatures2.shape()[0],infoFeatures2.shape()[1]);
        MultiArray<2, double> features_test(shpFeatures2);
        readHDF5(infoFeatures2, features_test);

        std::string path4="reference_prediction";
        HDF5ImportInfo infoRef(filename.c_str(),path4.c_str() );
        MultiArrayShape<2>::type shpRef(infoRef.shape()[0],infoRef.shape()[1]);
        MultiArray<2, double> ref(shpRef);
        readHDF5(infoRef, ref);

        std::cout << "Data loaded " << features_test.shape(0) <<" , " << features_test.shape(1) << "\n";

        //vigra::RegressionSplit rsplit;
        //vigra::rf::split::HoughSplitTest rsplit;
        DepthAndSizeStopping stop(20,5);
        rf::visitors::RandomForestProgressVisitor progressvisit;
        rf::visitors::VisitorBase defaultvisit;


        RandomForestOptions options;
        options.tree_count(10).features_per_node(RF_ALL).sample_with_replacement(false).samples_per_tree(0.9);

        RandomForest<double,RegressionTag> rf(options);


        vigra::TestRegressionSplit rsplit;
        vigra::RandomMT19937 rgn(42);
        rf.learn(features, labels,create_visitor(defaultvisit),rsplit,stop,rgn);

        std::cout << " Training done !\n";


        MultiArrayShape<2>::type shpRes(features_test.shape(0),rf.ext_param_.class_count_);
        MultiArray<2, double> result(shpRes);


        rf.predictRaw(features_test,result);

        //CHECK THE SHAPE OF THE RESULT IS CORRECT
        shouldEqual(result.shape(0),ref.shape(0));
        shouldEqual(result.shape(1),ref.shape(1));

        //CHECK THE RESULT DID NOT CHANGE DURING REFACTORING
        shouldEqual(result,ref);

    }


    /**
     * ND regression test. Checks that the results does not differ from
     * a reference which was manually checked. The paremeters of the rf should be fixed.
     */
    void RegressionForestTestND()
    {


        std::cout << "Entering the Regression Forest test ND \n";
        std::string filename="data/regressionND_noise.h5";

        std::string path1="features_train";
        HDF5ImportInfo infoFeatures(filename.c_str(),path1.c_str() );
        MultiArrayShape<2>::type shpFeatures(infoFeatures.shape()[0],infoFeatures.shape()[1]);
        MultiArray<2, double> features(shpFeatures);
        readHDF5(infoFeatures, features);

        std::string path2="labels_train";
        HDF5ImportInfo infoLabels(filename.c_str(),path2.c_str() );
        MultiArrayShape<2>::type shpLabels(infoLabels.shape()[0],infoLabels.shape()[1]);
        MultiArray<2, double> labels(shpLabels);
        readHDF5(infoLabels, labels);

        std::string path3="features_test";
        HDF5ImportInfo infoFeatures2(filename.c_str(),path3.c_str() );
        MultiArrayShape<2>::type shpFeatures2(infoFeatures2.shape()[0],infoFeatures2.shape()[1]);
        MultiArray<2, double> features_test(shpFeatures2);
        readHDF5(infoFeatures2, features_test);

        std::string filen4="data/regressionND_noise.h5";
        std::string path4="reference_prediction";
        HDF5ImportInfo infoRef(filename.c_str(),path4.c_str() );
        MultiArrayShape<2>::type shpRef(infoRef.shape()[0],infoRef.shape()[1]);
        MultiArray<2, double> ref(shpRef);
        readHDF5(infoRef, ref);

        std::cout << "Data loaded " << features_test.shape(0) <<" , " << features_test.shape(1) << "\n";

        //vigra::RegressionSplit rsplit;
        //vigra::rf::split::HoughSplitTest rsplit;
        DepthAndSizeStopping stop(20,5);
        rf::visitors::RandomForestProgressVisitor progressvisit;
        rf::visitors::VisitorBase defaultvisit;


        RandomForestOptions options;
        options.tree_count(10).features_per_node(RF_ALL).sample_with_replacement(false).samples_per_tree(0.9);

        RandomForest<double,RegressionTag> rf(options);


        vigra::TestRegressionSplit rsplit;
        vigra::RandomMT19937 rgn(42);
        rf.learn(features, labels,create_visitor(defaultvisit),rsplit,stop,rgn);

        std::cout << " Training done !\n";


        MultiArrayShape<2>::type shpRes(features_test.shape(0),rf.ext_param_.class_count_);
        MultiArray<2, double> result(shpRes);


        rf.predictRaw(features_test,result);

        //CHECK THE SHAPE OF THE RESULT IS CORRECT
        shouldEqual(result.shape(0),ref.shape(0));
        shouldEqual(result.shape(1),ref.shape(1));

        //CHECK THE RESULT DID NOT CHANGE DURING REFACTORING
//        for (int i=0;i<result.shape(0);i++)
//        {
//            std::cerr<< features_test(i,0) <<" " << result(i,0) <<" " << ref(i,0) << "\n";
//        }

        shouldEqual(result,ref);

    }

};

struct RegressionForestTestSuite: public vigra::test_suite
{
    RegressionForestTestSuite() :
        vigra::test_suite("RegressionForestTestSuite")
    {
        add(testCase( &RegressionForestTest::RegressionForestTest1D));
        add(testCase( &RegressionForestTest::RegressionForestTestND));
    }
};

#if CLASSIFIER_TEST
int main(int argc, char ** argv)
{
    RegressionForestTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;
    return (failed != 0);
}
#endif

