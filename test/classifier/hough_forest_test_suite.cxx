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
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
//We need to undefine NDEBUG so that we have TIC, TOC available!
#undef NDEBUG
#include <vigra/timing.hxx>
//#include "../vigranumpy/private/object_features/HoughForest.hpp" //FIXME: this should be moved to a more appropriate location
#include <vigra/random_forest/hough_forest.hxx>
#include <vigra/impex.hxx>

/// Include here the files for the random regresion forest

USETICTOC
;

using namespace vigra;

struct HoughForestTest
{


    //


    //Hough Forest Data

    MultiArray<2, double> features_hf_train;
    MultiArray<2, double> response_hf_train;

    MultiArray<2, double> features_hf_test;
    MultiArray<2, double> offset_hf_test;
    MultiArray<1, double> shpIm_hf_test;


    HoughForestTest()
    {
        this->setUpDataHF();
    }


    void setUpDataHF()
    {
        std::string filen="data/HF_data_cells_train.h5";
        std::string path1="features";
        HDF5ImportInfo infoFeatures(filen.c_str(),path1.c_str() );
        MultiArrayShape<2>::type shpFeatures(infoFeatures.shape()[0],infoFeatures.shape()[1]);
        features_hf_train.reshape(shpFeatures);
        readHDF5(infoFeatures, features_hf_train);



        std::string path2="response";
        HDF5ImportInfo infoLabels(filen.c_str(), path2.c_str());
        MultiArrayShape<2>::type shpLabels(infoLabels.shape()[0],infoLabels.shape()[1]);
                response_hf_train.reshape(shpLabels);
                readHDF5(infoLabels, response_hf_train);


        should(features_hf_train.shape(0)==response_hf_train.shape(0));


        std::string filen2="data/HF_data_cells_test.h5";
        HDF5ImportInfo infoFeatures2(filen2.c_str(),path1.c_str() );
        MultiArrayShape<2>::type shpFeatures2(infoFeatures2.shape()[0],infoFeatures2.shape()[1]);
        features_hf_test.reshape(shpFeatures2);
        readHDF5(infoFeatures2, features_hf_test);

        std::string path3="offsets";
        HDF5ImportInfo infoOffsets(filen2.c_str(),path3.c_str() );
        MultiArrayShape<2>::type shpOffsets(infoOffsets.shape()[0],infoOffsets.shape()[1]);
        offset_hf_test.reshape(shpOffsets);
        readHDF5(infoOffsets, offset_hf_test);

        std::string path4="imshape";
        HDF5ImportInfo infoImshape(filen2.c_str(),path4.c_str() );
        MultiArrayShape<1>::type shpImshape(infoImshape.shape()[0]);
        shpIm_hf_test.reshape(shpImshape);
        readHDF5(infoImshape, shpIm_hf_test);

    }


    /**
     * Redundant
     * ONLY Check that the base class of the Hough Forest Run trough
     */
    void HoughForestBaseTest()
    {
        vigra::rf::visitors::IndexVisitorTraining visitorLearning;

        rf::visitors::RandomForestProgressVisitor progressvisit;

        DepthAndSizeStopping stop(10,20);
        vigra::RandomForest<double, HoughTag> RF(
                RandomForestOptions().tree_count(5).sample_with_replacement(
                        true));

        RF.learn(features_hf_train, response_hf_train, create_visitor(visitorLearning),
                rf::split::HoughSplitGini (), stop,
                vigra::RandomMT19937(42));

        vigra::rf::visitors::IndexVisitorPrediction visitorPrediction;
        MultiArrayShape<2>::type sh(features_hf_test.shape(0), 2);
        MultiArray<2, double> res(sh);
        RF.predictProbabilities(features_hf_test, res, rf_default(),visitorPrediction);
    }




    /**
     *
     * Check that the prediction of the particular random forest is equal to the result serialized in the file\
     *
     * give the possibility to serialize current result as new reference
     */
    void HoughForestTestInfer(Hough_Forest<double,double,UnstridedArrayTag>& HF,std::string basenamereference, bool make_reference= false)
    {


        vigra::rf::visitors::IndexVisitorPrediction visitorPrediction;
        MultiArrayShape<2>::type sh(shpIm_hf_test(1), shpIm_hf_test(0));
        MultiArray<2, double> result(sh);
        HF.predictOnImage(features_hf_test,offset_hf_test,shpIm_hf_test(1),shpIm_hf_test(0),30,result);


        if(make_reference)
        {
        //FIXME: we export as png and hdf5 since exportImage seems to normlize automatically the output which
        // I would like to avoid
            ImageExportInfo infoex((basenamereference+ "new.png").c_str());
            exportImage(srcImageRange(result),infoex);
            writeHDF5((basenamereference+ "new.h5").c_str(),"reference", result);
        }

        HDF5ImportInfo inforef((basenamereference+ ".h5").c_str(),"reference");
        MultiArrayShape<2>::type shpref(inforef.shape()[0],inforef.shape()[1]);
        MultiArray<2, double> reference(shpref);
        readHDF5(inforef, reference);
        shouldEqualSequenceTolerance(reference.begin(), reference.end(),
                result.begin(),0.01);

    }

    void HoughForestTestOG()
    {
        bool make_reference=false;
        RandomForestOptions options;
        options.tree_count(5);
//        options.use_stratification(RF_EQUAL);
//        options.samples_per_tree(0.9);
//        options.features_per_node(mtry)
        Hough_Forest<double,double,UnstridedArrayTag> HF(options,10, 20,42);

        HF.learnOrthogonalGini(features_hf_train,response_hf_train);

        std::string basenamereference="data/hf_cells_reference_OG";
        this->HoughForestTestInfer(HF,basenamereference,make_reference);

    }

    void HoughForestTestOE()
    {
        RandomForestOptions options;
        options.tree_count(5);
//        options.use_stratification(RF_EQUAL);
//        options.samples_per_tree(0.9);
//        options.features_per_node(mtry)
        Hough_Forest<double,double,UnstridedArrayTag> HF(options,10, 20,42);
        HF.learnOrthogonalEntropy(features_hf_train,response_hf_train);
        std::string basenamereference="data/hf_cells_reference_OE";
        this->HoughForestTestInfer(HF,basenamereference);
    }



    /**
     * Hough Forest advanced test Random Entropy split functor
     */
    void HoughForestTestRE()
    {

        RandomForestOptions options;
        options.tree_count(2);
//        options.use_stratification(RF_EQUAL);
        options.samples_per_tree(0.5);
//        options.features_per_node(mtry)
        Hough_Forest<double,double,UnstridedArrayTag> HF(options,3, 50,42);

        HF.learnRandomEntropy(features_hf_train,response_hf_train);
        std::string basenamereference="data/hf_cells_reference_RE";
        this->HoughForestTestInfer(HF,basenamereference);
    }


    /**
     * Hough Forest advanced test Random Gini split functor
     */
    void HoughForestTestRG()
    {

        RandomForestOptions options;
        options.tree_count(2);
//        options.use_stratification(RF_EQUAL);
        options.samples_per_tree(0.5);
//        options.features_per_node(mtry)
        Hough_Forest<double,double,UnstridedArrayTag> HF(options,3, 50,42);

        HF.learnRandomGini(features_hf_train,response_hf_train);
        std::string basenamereference="data/hf_cells_reference_RG";
        this->HoughForestTestInfer(HF,basenamereference);
    }


    /**
     * Test for serialization deserialization of the visitors of the HF, checks that the result is not corrupted during serialization
     */
    void SaveLoadVisitorsHFTest()
    {
        vigra::rf::visitors::IndexVisitorTraining visitorLearning;



        double rawfeatures[] =
        { 2, 0, 1, 1, 0, 0, 0,
          1, 0, 1, 1, 0, 0, 0

        };

        double rawlabels[] =
        { 1, 0, 1, 1, 0, 0, 0,
          4, 1, 5, 5, 1, 1, 1,
          4, 2, 2, 2, 1, 1, 1
        };

        typedef MultiArrayShape<2>::type Shp;
            MultiArrayView<2, double> features(Shp(7, 2), rawfeatures);
            MultiArrayView<2, double> labels(Shp(7, 3), rawlabels);

        std::cout << "The labels are:  " << std::endl << labels << std::endl;
        std::cout << "The features are: " << std::endl << features << std::endl;


        vigra::RandomForest<double, HoughTag> RF(
                RandomForestOptions().tree_count(20).sample_with_replacement(
                        false));


        MultiArray<2, double> predicted(Shp(7, 2));

        rf::visitors::RandomForestProgressVisitor progressvisit;
        RF.learn(features, labels, create_visitor(visitorLearning,progressvisit),
                rf::split::HoughSplitRandomEntropy (), rf_default(),
                vigra::RandomMT19937(42));

        vigra::rf::visitors::IndexVisitorPrediction visitorPrediction;


        RF.predictProbabilities(
                        features, predicted, rf_default(), visitorPrediction);


        std::cout << "Predicted Probabilitites" << std::endl;

        std::cout<<predicted<< std::endl;

        std::cout << "saving the visitor" << std::endl;

        visitorLearning.save("test2.h5", "test/res");

        vigra::rf::visitors::IndexVisitorTraining visitorLearning2;

        std::cout << "loading the visitor" << std::endl;
        visitorLearning2.load("test2.h5", "test/res");


        should(visitorLearning2.mapping==visitorLearning.mapping);

        should(visitorLearning2.mapping_to_prob==visitorLearning.mapping_to_prob);


        vigra::rf::visitors::IndexVisitorPrediction visitorPrediction2;



        visitorPrediction.save("test2.h5","test/prediction");
        visitorPrediction2.load("test2.h5","test/prediction");

        std::cout<< "First visitor" <<std::endl;

        visitorPrediction.deepFout();

        std::cout<< "Second visitor" <<std::endl;
        visitorPrediction2.deepFout();

        should(visitorPrediction.mapping==visitorPrediction2.mapping);

    }

    /**
     * Other minimal test for the serialization deserialization of visitor learning and visitor prediction
     */
    void SaveLoadVisitorHFTest2()
    {
        vigra::rf::visitors::IndexVisitorTraining visitorLearning;



        double rawfeatures[] =
        { 2, 0, 1, 1, 0, 0, 0,
          1, 0, 1, 1, 0, 0, 0

        };

        double rawlabels[] =
        { 1, 0, 1, 1, 0, 0, 0,
          4, 1, 5, 5, 1, 1, 1,
          4, 2, 2, 2, 1, 1, 1
        };

        typedef MultiArrayShape<2>::type Shp;
            MultiArrayView<2, double> features(Shp(7, 2), rawfeatures);
            MultiArrayView<2, double> labels(Shp(7, 3), rawlabels);

        std::cout << "The labels are:  " << std::endl << labels << std::endl;
        std::cout << "The features are: " << std::endl << features << std::endl;


        vigra::RandomForest<double, HoughTag> RF(
                RandomForestOptions().tree_count(20).sample_with_replacement(
                        false));


        MultiArray<2, double> predicted(Shp(7, 2));

        rf::visitors::RandomForestProgressVisitor progressvisit;
        RF.learn(features, labels, create_visitor(visitorLearning,progressvisit),
                rf::split::HoughSplitGini (), rf_default(),
                vigra::RandomMT19937(42));

        vigra::rf::visitors::IndexVisitorPrediction visitorPrediction;


        RF.predictProbabilities(
                        features, predicted, rf_default(), visitorPrediction);


        std::cout << "Predicted Probabilitites" << std::endl;

        std::cout<<predicted<< std::endl;

        std::cout << "saving the visitor" << std::endl;

        visitorLearning.save("test2.h5", "test/res");

        vigra::rf::visitors::IndexVisitorTraining visitorLearning2;

        std::cout << "loading the visitor" << std::endl;
        visitorLearning2.load("test2.h5", "test/res");


        should(visitorLearning2.mapping==visitorLearning.mapping);
//        visitorLearning2.mapping_to_prob[]==visitorLearning.mapping_to_prob);

//        shouldEqual(visitorLearning2.mapping_to_prob,visitorLearning.mapping_to_prob);

        typename vigra::rf::visitors::IndexVisitorTraining::KeyIt it;
        for (it = visitorLearning.mapping.begin(); it != visitorLearning.mapping.end(); it++)
        {
            should(visitorLearning.mapping.count(it->first)==1);
            should(visitorLearning2.mapping.count(it->first)==1);

            should(visitorLearning.mapping_to_prob.count(it->first)==1);
            should(visitorLearning2.mapping_to_prob.count(it->first)==1);
            shouldEqual(visitorLearning2.mapping_to_prob[it->first],visitorLearning.mapping_to_prob[it->first]);

        }

        vigra::rf::visitors::IndexVisitorPrediction visitorPrediction2;



        visitorPrediction.save("test2.h5","test/prediction");
        visitorPrediction2.load("test2.h5","test/prediction");

        should(visitorPrediction.mapping==visitorPrediction2.mapping);

    }


};

struct HoughForestTestSuite: public vigra::test_suite
{
    HoughForestTestSuite() :
        vigra::test_suite("HoughForestTestSuite")
    {

        //add(testCase( &HoughForestTest::SaveLoadVisitorTest));

        add(testCase( &HoughForestTest::SaveLoadVisitorsHFTest));
        add(testCase( &HoughForestTest::SaveLoadVisitorHFTest2));

        add(testCase( &HoughForestTest::HoughForestBaseTest));


        add(testCase( &HoughForestTest::HoughForestTestRE));
        add(testCase( &HoughForestTest::HoughForestTestRG));
        add(testCase( &HoughForestTest::HoughForestTestOG));
        add(testCase( &HoughForestTest::HoughForestTestOE));




    }
};

#if CLASSIFIER_TEST
int main(int argc, char ** argv)
{
    HoughForestTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;
    return (failed != 0);
}
#endif

