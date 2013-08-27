/**
 * This file implements the HoughForest prediction method
 *
 *
 *
 *
 */

#include <vigra/random_forest.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/random_forest/rf_more_visitor.hxx>

#include <set>
#include <cmath>
#include <memory>
#include <limits>

//file exists
#include <sys/stat.h>

#include <cstdio>
//#include <vigra/numpy_array.hxx>




#define PIG 3.14159265



namespace vigra
{


int getBin(double dy,double dx,const int bins)
{
	if (dy!=0) dy=-dy;

	double angle= std::atan2(dy,dx)*180.0/PIG;
	if (angle<0) angle+=180;
	double dt= 180/bins;


	int bin = std::floor(angle / dt);

	//std::cerr << "dx "<< dx << ", dy "<< dy <<" , angle "<< angle << " , " << bin << std::endl;


	//printf("dx is = %f, dy is = %f, angle is = %f, bin is =% d \n",dx,dy,angle,bin);
	if (bin==bins)
	   bin=0;

	return bin;

}



template<class LabelType, class T =float, class C=StridedArrayTag>
class Hough_Forest: public RandomForest<LabelType, HoughTag>
{
public:
	RandomForestOptions options_;

	typedef std::pair<int, int> LeafType;
	typedef int SampleType;

	vigra::rf::visitors::IndexVisitorTraining visitor_learning;
	vigra::rf::visitors::IndexVisitorPrediction visitor_prediction;

	int dim_labels;
	int dim_features;

	int ntr_samples;
	int nte_samples;

	DepthAndSizeStopping stop;

	MultiArray<2,LabelType> tr_labels;
	//NumpyArray<2, T> tr_labels;

	int seed;

	typedef vigra::RandomMT19937 RandomNumberGenerator;

	//FIXME: This is used in pythonHFImportFromHDF5 as an ack to initilize a pointer, it should be probably remove
	Hough_Forest() : dim_labels(0),	dim_features(0),ntr_samples( 0),nte_samples( 0),seed(-1)
	{
	}

	//FIXME: All this constructors should be nested instead of bein writte one after the other
	Hough_Forest(RandomForestOptions options, int max_depth, int min_size) :
		RandomForest<LabelType, HoughTag> (options), options_(options), stop(
				max_depth, min_size),seed(-1)
	{
		dim_labels = 0;
		dim_features = 0;
		ntr_samples = 0;
		nte_samples = 0;
	}

	Hough_Forest(RandomForestOptions options, int max_depth, int min_size, int seed_) :
		RandomForest<LabelType, HoughTag> (options), options_(options), stop(
				max_depth, min_size), seed(seed_)
	{
		dim_labels = 0;
		dim_features = 0;
		ntr_samples = 0;
		nte_samples = 0;
	}


	Hough_Forest(RandomForestOptions options) :
		RandomForest<LabelType, HoughTag> (options), options_(options),seed(-1)
	{
		dim_labels = 0;
		dim_features = 0;
		ntr_samples = 0;
		nte_samples = 0;
	}

	Hough_Forest(int treeCount) :
		RandomForest<LabelType, HoughTag> (
				vigra::RandomForestOptions().tree_count(treeCount)),seed(-1)
	{
		dim_labels = 0;
		dim_features = 0;
		ntr_samples = 0;
		nte_samples = 0;
	}



	//FIXME: Random Gini and Random Entropy work in general a bit better BUT the implementation makes them much slower
	// the implementation of the split functor should be completely reworked
	void learnRandomGini(const MultiArrayView<2,LabelType,C>& train_data, const MultiArrayView<2,LabelType,C>& train_labels)
	{
		ntr_samples = train_data.shape(0);
		dim_labels = train_labels.shape(1);
		dim_features = train_data.shape(1);

		tr_labels = train_labels;

		rf::visitors::RandomForestProgressVisitor progressvisit;

		vigra::rf::split::HoughSplitRandomGini rsplit;


		if (seed<0)
		{
		this->learn(train_data, train_labels,
				create_visitor((*this).visitor_learning, progressvisit),
				rsplit, stop);
		}
		else
		{
			RandomNumberGenerator rng(seed);
			this->learn(train_data, train_labels,
					create_visitor((*this).visitor_learning, progressvisit),
					rsplit, stop,rng);

		}


	}


	void learnRandomEntropy(const MultiArrayView<2,LabelType,C>& train_data, const MultiArrayView<2,LabelType,C>& train_labels)
	{
		ntr_samples = train_data.shape(0);
		dim_labels = train_labels.shape(1);
		dim_features = train_data.shape(1);

		tr_labels = train_labels;

		rf::visitors::RandomForestProgressVisitor progressvisit;

		vigra::rf::split::HoughSplitRandomEntropy rsplit;


		if (seed<0)
		{
		this->learn(train_data, train_labels,
				create_visitor((*this).visitor_learning, progressvisit),
				rsplit, stop);
		}
		else
		{
			RandomNumberGenerator rng(seed);
			this->learn(train_data, train_labels,
					create_visitor((*this).visitor_learning, progressvisit),
					rsplit, stop,rng);

		}

	}


	void learnOrthogonalEntropy(const MultiArrayView<2,LabelType,C>& train_data, const MultiArrayView<2,LabelType,C>& train_labels)
	{
		ntr_samples = train_data.shape(0);
		dim_labels = train_labels.shape(1);
		dim_features = train_data.shape(1);

		tr_labels = train_labels;

		rf::visitors::RandomForestProgressVisitor progressvisit;

		vigra::rf::split::HoughSplitEntropy rsplit;


		if (seed<0)
		{
		this->learn(train_data, train_labels,
				create_visitor((*this).visitor_learning, progressvisit),
				rsplit, stop);
		}
		else
		{
			RandomNumberGenerator rng(seed);
			this->learn(train_data, train_labels,
					create_visitor((*this).visitor_learning, progressvisit),
					rsplit, stop,rng);

		}

	}


	void learnOrthogonalGini(const MultiArrayView<2,LabelType,C>& train_data, const MultiArrayView<2,LabelType,C>& train_labels)
	{
		ntr_samples = train_data.shape(0);
		dim_labels = train_labels.shape(1);
		dim_features = train_data.shape(1);

		tr_labels = train_labels;

		rf::visitors::RandomForestProgressVisitor progressvisit;

		vigra::rf::split::HoughSplitGini rsplit;


		if (seed<0)
		{
		this->learn(train_data, train_labels,
				create_visitor((*this).visitor_learning, progressvisit),
				rsplit, stop);
		}
		else
		{
			RandomNumberGenerator rng(seed);
			this->learn(train_data, train_labels,
					create_visitor((*this).visitor_learning, progressvisit),
					rsplit, stop,rng);
            visitor_learning.save("OGlearnvisit_new.h5", "/learn");

		}

	}

	template<class result, class A>
	void predict(const A& test_data, result res)
	{
		visitor_prediction.clear();

		return this->predictProbabilities(
				test_data, res, rf_default(), (*this).visitor_prediction);
	}



	/**
	 * This is the function that mekes the prediction of the hough forests
	 *
	 * @param test_data Must be a set of patches for which we want to predict reshaped as a matrix N*nfeat (can be all patches in the image)
	 * @param patch_centers (must store the centre of each patch N*2
	 * @param imgwidth width of the image we want to predict
	 * @param imgheight heigth of the image we want to predict
	 * @param factor gain factor of the image to predict (contrast)
	 * @param result *initialized result
	 */

	template<class T2, class C2, class T3, class C3>
	void predictOnImage(MultiArrayView<2, T2, C2> test_data, MultiArrayView<2, T2,
			C2> patch_centers, int imgwidth, int imgheight,int factor,
			MultiArrayView<2, T3, C3> result)
	{

		typedef std::vector<double> Point;

		//FIXME: currently the algorithm assumes that result shape is imgheight,imgwidth
		// problably it is redundand to pass this two parameters from outside
		if (result.shape(0) != imgheight or result.shape(1) != imgwidth)
			throw std::runtime_error("The result has a wrong shape!, shoudl be (imgheight,imgwidth)");

		if (patch_centers.shape(0) != test_data.shape(0))
			throw std::runtime_error("shape mismatch");

		//FIXME in the far future, Generalize this function to multilabel Hough Forest
		MultiArrayShape<2>::type sh(test_data.shape(0), 2);

		MultiArray<2, T> dummy(sh);

		this->predict(test_data, dummy);


		int count = 0;
		int inscount = 0;

		//Loop on the samples

		for (int i = 0; i < test_data.shape(0); i++)
		{
			std::vector<LeafType>& leafs =
					(this->visitor_prediction).mapping[i];

			int ox = patch_centers(i, 0); //patch centers offsets vectors
			int oy = patch_centers(i, 1);

			double nleafs = leafs.size(); // number of leafs in the associated node
			//std::cout << "N leafs " << nleafs << std::endl;


			for (std::vector<LeafType>::iterator it = leafs.begin(); it
					!= leafs.end(); it++)
			{
				std::vector<Point>& current_points =
						(this->visitor_learning).mapping[*it];

				double prob = (this->visitor_learning).mapping_to_prob[*it];
				//std::cout<<"the prob is " << prob << std::endl;

				std::vector<Point>::iterator ii;

				double npoints = current_points.size();

				for (ii = current_points.begin(); ii != current_points.end(); ++ii)
				{

					int x = (*ii)[0] + ox;
					int y = (*ii)[1] + oy;

					//std::cout<< "The offset is " <<x << " " << y << std::endl;

					//std::cout<< " ( " << x  << ", " << y << " ) " ;

					if ((0 <= x) & (x < imgwidth) & (0 <= y) & (y < imgheight))
					{
						//res(y,x)+=prob/npoints/nleafs;
						//res(y,x)+=prob/npoints/nleafs*100;

						//if (prob > 1)
						//	throw std::runtime_error("prob>1");

						//if (prob < 0)
						//	throw std::runtime_error("prob<0");
						//std::cerr << x  << ", ";

						result(y, x) += factor * prob / (npoints * nleafs +1e-42);

						inscount++;
					}

					//else
					//{
					//std::cout << "outside" << x << " " << y << std::endl;
					//	++count;
					//}
				}
			}
		}
		//std::cout << "the outside vote count is " << count << std::endl;
		//std::cout << "the inside vote count is " << inscount << std::endl;
	}


	//FIXME: REMOVE
	template<class T2, class C2, class T3, class C3>
	void predictOnImageWithAngle(MultiArrayView<2, T2, C2> test_data, MultiArrayView<2, T2,
			C2> patch_centers, int imgwidth, int imgheight,int bins, int factor,
			MultiArrayView<3, T3, C3> result)
	{

		typedef std::vector<double> Point;

		//if (result.shape(0) != imheight or result.shape(1) != imwidth)
		//	throw std::runtime_error("the result has a wrong shape!");

		//if (patch_centers.shape(0) != test_data.shape(0))
		//	throw std::runtime_error("shape mismatch");

		//FIXME Generalize to multilabel fix Numpy array
		MultiArrayShape<2>::type sh(test_data.shape(0), 2);

		MultiArray<2, T> dummy(sh);

		this->predict(test_data, dummy);

		int count = 0;
		int inscount = 0;

		//Loop on the samples

		for (int i = 0; i < test_data.shape(0); i++)
		{
			std::vector<LeafType>& leafs =
					(this->visitor_prediction).mapping[i];

			int ox = patch_centers(i, 0); //patch centers offsets vectors
			int oy = patch_centers(i, 1);

			double nleafs = leafs.size(); // number of leafs in the associated node
			//std::cout << "N leafs " << nleafs << std::endl;


			for (std::vector<LeafType>::iterator it = leafs.begin(); it
					!= leafs.end(); it++)
			{
				std::vector<Point>& current_points =
						(this->visitor_learning).mapping[*it];

				double prob = (this->visitor_learning).mapping_to_prob[*it];
				//std::cout<<"the prob is " << prob << std::endl;

				std::vector<Point>::iterator ii;

				double npoints = current_points.size();

				for (ii = current_points.begin(); ii != current_points.end(); ++ii)
				{

					int x = (*ii)[0] + ox;
					int y = (*ii)[1] + oy;
					double dx=(*ii)[0];
					double dy=(*ii)[1];

					int bin=getBin(dy,dx,bins);

					//std::cout<< "The offset is " <<x << " " << y << std::endl;


					if ((0 <= x) & (x < imgwidth) & (0 <= y) & (y < imgheight))
					{
						//res(y,x)+=prob/npoints/nleafs;
						//res(y,x)+=prob/npoints/nleafs*100;

						//if (prob > 1)
						//	throw std::runtime_error("prob>1");

						//if (prob < 0)
						//	throw std::runtime_error("prob<0");
						//std::cerr << x  << ", ";


						double vote=factor * prob /(npoints * nleafs);


						//std::cerr<< " ( " << x  << ", " << y << " ) " << "bin "<< bin <<std::endl;
						result(y, x,bin) +=vote;

						//addDiagonal(result,y,x,factor * prob / (npoints * nleafs),1.0);


						inscount++;
					}





				}
			}
		}
		//std::cout << "the outside vote count is " << count << std::endl;
		//std::cout << "the inside vote count is " << inscount << std::endl;
	}

};

#ifdef HasHDF5
bool FileExists(std::string strFilename)
{
	struct stat stFileInfo;
	bool blnReturn;
	int intStat;

	// Attempt to get the file attributes
	intStat = stat(strFilename.c_str(), &stFileInfo);
	if (intStat == 0)
	{
		// We were able to get the file attributes
		// so the file obviously exists.
		blnReturn = true;
	}
	else
	{
		// We were not able to get the file attributes.
		// This may mean that we don't have permission to
		// access the folder which contains this file. If you
		// need to do that level of checking, lookup the
		// return values of stat which will give you
		// more details on why stat failed.
		blnReturn = false;
	}

	return (blnReturn);
}

template<class LabelType, class T , class C >
bool
hf_import_HDF5(Hough_Forest<LabelType,T,C> &Hf,
		std::string fname)
{
	//const RandomForest<float> & xxx = static_cast<const RandomForest<float> &>(Hf);

	std::string pathname = "forest";


	if(!FileExists(fname))
	throw std::runtime_error("File  does not exists!");


	HDF5ImportInfo info(fname.c_str(),"parameters/tr_labels");

	//FIXME could this crash??
	MultiArrayShape<2>::type shape(info.shapeOfDimension(0), info.shapeOfDimension(1));

	//This is the variant with the numpy Array
	//NumpyArray<2,float> temp(shape);


	//readHDF5(info,temp);
	//std::cout<< "HERE" << Hf.tr_labels.shape() << std::endl;
	//Hf.tr_labels=temp;


	//This is the variant with the numpy Array
	//(Hf.tr_labels).reshapeIfEmpty(shape,
	//		"Output array has wrong dimensions.");

	Hf.tr_labels.reshape(shape);
	readHDF5(info,Hf.tr_labels);

	rf_import_HDF5(Hf,fname,pathname);

	Hf.visitor_learning.load(fname,"parameters/vLearning");

	return 1;
}

template<class LabelType, class T , class C >
bool
hf_saveToHDF5( Hough_Forest<LabelType,T,C> &Hf, std::string fname)
{



	if(FileExists(fname))
	remove( fname.c_str() );

	if(FileExists(fname))
	throw std::runtime_error("File exists!");



	rf_export_HDF5(Hf,
			fname,
			"forest");




	Hf.visitor_learning.save(fname,"parameters/vLearning");



	writeHDF5(fname.c_str(),"parameters/tr_labels",Hf.tr_labels);

	return 1;
}

#endif // HasHDF5

} // namespace vigra


