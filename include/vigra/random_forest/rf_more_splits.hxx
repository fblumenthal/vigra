/************************************************************************/
/*                                                                      */
/*        Copyright 2011-2013 by Luca Fiaschi                           */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#ifndef VIGRA_RANDOM_FOREST_MORE_SPLITS_HXX
#define VIGRA_RANDOM_FOREST_MORE_SPLITS_HXX

#include "rf_split.hxx"
#include "rf_common.hxx"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

namespace vigra
{
namespace rf
{

namespace split
{

using namespace std;
/** This Functor chooses a random value of a column
 */
template<class LossTag>
class BestSplitOfRandomEnsemble
{
public:

	ArrayVector<double> class_weights_;
	ArrayVector<double> bestCurrentCounts[2];
	double min_gini_;
	ptrdiff_t min_index_;
	double min_threshold_;
	ProblemSpec<> ext_param_;
	typedef RandomMT19937 Random_t;
	Random_t random;

	BestSplitOfRandomEnsemble()
	{
	}

	template<class T>
	BestSplitOfRandomEnsemble(ProblemSpec<T> const & ext) :
		class_weights_(ext.class_weights_), ext_param_(ext), random(RandomSeed)
	{
		bestCurrentCounts[0].resize(ext.class_count_);
		bestCurrentCounts[1].resize(ext.class_count_);
	}

	template<class T>
	BestSplitOfRandomEnsemble(ProblemSpec<T> const & ext, Random_t & random_) :
		class_weights_(ext.class_weights_), ext_param_(ext), random(random_)
	{
		bestCurrentCounts[0].resize(ext.class_count_);
		bestCurrentCounts[1].resize(ext.class_count_);
	}

	template<class T>
	void set_external_parameters(ProblemSpec<T> const & ext)
	{
		class_weights_ = ext.class_weights_;
		ext_param_ = ext;
		bestCurrentCounts[0].resize(ext.class_count_);
		bestCurrentCounts[1].resize(ext.class_count_);
	}

	template<class DataSourceF_t, class DataSource_t, class I_Iter, class Array>
	void operator()(DataSourceF_t const & column, DataSource_t const & labels,
			I_Iter & begin, I_Iter & end, Array const & region_response)
	{
		std::sort(begin, end,
				SortSamplesByDimensions<DataSourceF_t> (column, 0));
		typedef typename LossTraits<LossTag, DataSource_t>::type
				RandomSearchLoss;
		RandomSearchLoss left(labels, ext_param_); //initialize the left and right part
		RandomSearchLoss right(labels, ext_param_);
		//right.init(begin, end, region_response);

		int number_splits_to_try = 1;
		ArrayVector<int> ensamble_splits(number_splits_to_try);
		for (int i = 0; i < number_splits_to_try; i++)
		{
			ensamble_splits.push_back(*begin + random.uniformInt(end - begin));
			//std::cout << "split at the index " << *begin + random.uniformInt(end - begin) << std::endl;
		}

		min_gini_ = NumericTraits<double>::max();
		min_index_ = ensamble_splits[0]; //initialize min index with the first one

		double temp_gini;
		for (int i = 0; i < number_splits_to_try; i++)
		{

			temp_gini = right.init(begin + ensamble_splits[i], end,
					region_response);
			if (min_gini_ > temp_gini)
			{
				min_gini_ = temp_gini;
				min_index_ = ensamble_splits[i];
			}
		}

	}

	template<class DataSource_t, class Iter, class Array>
	double loss_of_region(DataSource_t const & labels, Iter & begin,
			Iter & end, Array const & region_response) const
	{
		typedef typename LossTraits<LossTag, DataSource_t>::type LineSearchLoss;
		LineSearchLoss region_loss(labels, ext_param_);
		return region_loss.init(begin, end, region_response);
	}

};

typedef ThresholdSplit<BestSplitOfRandomEnsemble<GiniCriterion> >
		ExtremelyRandomGiniSplit;



//Auxiliari structure to evaluate the splits
class Vals
{
public:
	int index;
	double value;
	bool operator <(const Vals &other) const
	{
		return (this->value < other.value);
	}
};

class SplitParams
{
public:
	double intercept;
	std::vector<double> weights;
	std::vector<int> columns; //the columns associated with the weights

	int ndimensions;
	bool isDefined;

	SplitParams(int ndimensions_)
	{
		isDefined = false;
		ndimensions = ndimensions_;
		intercept = 0.0;
		weights.resize(ndimensions, 0.0);
		columns.resize(ndimensions, 0.0);
	}

	template<class Container>
	SplitParams(int ndimensions_, const Container & splitColumns)
	{
		isDefined = false;
		columns.resize(ndimensions_, 0.0);
		ndimensions = ndimensions_;
		intercept = 0.0;
		weights.resize(ndimensions_, 0.0);

		for (int i = 0; i < ndimensions; i++)
		{
			columns[i] = splitColumns[i];
		}
	}
	void printOut()
	{
		std::cerr << "The intercept is " << intercept << std::endl;

		for (int i = 0; i < ndimensions; i++)
		{
			std::cerr << "The weights for the column " << columns[i] << "is "
					<< weights[i] << std::endl;

		}
	}

};




template<class CriterionFunctor1, class CriterionFunctor2, class Tag = HoughTag>
class RandomHyperplaneSplit: public SplitBase<Tag>
{
public:

	typedef SplitBase<Tag> SB;

	ArrayVector<Int32> splitColumns; //the columns to be splitted


	int bestSplitIndex;

	int nsplits2try;
	int ndimensions; //the number of dimensions to use

	int mode; //The mode of the current split;


	int bestSplitColumn() const
	{
		throw std::runtime_error("this function should never be called!!");
		return 0;
	}

	template<class T>
	void set_external_parameters(ProblemSpec<T> const & in)
	{
		SB::set_external_parameters(in);

		int featureCount_ = SB::ext_param_.column_count_;
		splitColumns.resize(featureCount_);
		for (int k = 0; k < featureCount_; ++k)
			splitColumns[k] = k;

		//FIXME now they are hard coded
		nsplits2try = 1000;
		ndimensions = 2;
	}

	template<class T, class C, class T2, class C2, class Region, class Random>
	int makeTerminalNode(const MultiArrayView<2, T, C>& features,
			const MultiArrayView<2, T2, C2>& labels, Region & region,
			Random randint)
	{

		std::vector<double> hist(2, 0.0);
		typename Region::IndexIterator ii;
		ii = region.begin_;
		int max_label = 0;
		int total = 0;
		while (ii != region.end_)
		{
			double current_label = labels(*ii, 0); //nb assumes

			if (current_label >= max_label)
				max_label = current_label;

			hist[current_label] += 1.0;
			total += 1.0;
			++ii;
		}

		hist.resize(max_label + 1);

		Node<e_ConstProbNode> ret(SB::t_data, SB::p_data);
		SB::node_ = ret;

		std::copy(hist.begin(), hist.end(), ret.prob_begin());

		detail::Normalise<Tag>::exec(ret.prob_begin(), ret.prob_end());

		ret.weights() = region.size();
		return e_ConstProbNode;
	}

	//find the best split among a certain number of non orthogonal splits
	template<class T, class C, class T2, class C2, class Region, class Random>
	int findBestSplit(MultiArrayView<2, T, C> features, MultiArrayView<2, T2,
			C2> labels, Region & region, ArrayVector<Region>& childRegions,
			Random & randint)
	{
		typedef typename Region::IndexIterator IndexIterator;
		if (region.size() == 0)
		{
			std::cerr
					<< "SplitFunctor::findBestSplit(): stackentry with 0 examples encountered\n"
						"continuing learning process....";
		}

		/*
		 //initialize a double random number generator
		 boost::mt19937 rng;
		 //FIXME seed the time
		 rng.seed(static_cast<unsigned int>(std::time(0)));
		 boost::uniform_real<double> u(-1.0, 1.0);
		 boost::variate_generator<boost::mt19937&, boost::uniform_real<double> > gen(rng, u);
		 */

		//choose which columns (feature use to define the hyperplane)
		/*for (int ii = 0; ii < ndimensions; ++ii)
		 {
		 std::swap(splitColumns[ii], splitColumns[ii + randint(
		 features.shape(1) - ii)]);
		 }
		 */

		//Choose the mode of the split 1 regression 0 classification
		//if (std::accumulate(region.classCounts().begin(),
		//		region.classCounts().end(), 0) != region.size())
		MultiArrayView <2, T, C>  bla=columnVector(labels, 0);; //FIXME: The class RandomForestClassCounter is prone to be buggy in combination with columnVector
		//indeed it columnVector can store a referece which is invalidated inside this block, that's why bla needs to be allocated outside
		{
			RandomForestClassCounter<MultiArrayView<2, T2, C2> , ArrayVector<
					double> > counter(bla,
					region.classCounts());

			std::for_each(region.begin(), region.end(), counter);
			region.classCountsIsValid = true;
		}

		mode = randint(2) ;
		if (region.classCounts()[0] + region.classCounts()[1] != region.size())
			throw std::runtime_error("How can it be?");
		//std::cerr << region.classCounts()[0] / double(region.size()) << std::endl;

		if (region.classCounts()[0] / double(region.size()) < 0.05)
		{
			//std::cerr << "to few examples negative to choose entropy" << std::endl;
			mode = 1;
		}



		//Node<i_HyperplaneNode> node(ndimensions, SB::t_data, SB::p_data);


		SplitParams test(ndimensions);

		if (optimizeTest(features, labels, test, region, childRegions, randint))
		{

			//std::cerr << "Best split found the regions are splitted in this way :" << std::endl;
			//region.printRange();
			//childRegions[0].printRange();
			//childRegions[1].printRange();

			/*
			std::cerr<< "child region 0 " << std::endl;
			for(int kk = 0; kk<childRegions[0].size();++kk)
			{
				std::cerr << rowVector(features, childRegions[0][kk]) << std::endl;
			}

			std::cerr<< "child region 1 " << std::endl;
						for(int kk = 0; kk<childRegions[1].size();++kk)
						{
							std::cerr << rowVector(features, childRegions[1][kk]) << std::endl;
						}
			 */
			//SB::node_.topology_=node.topology_;
			//SB::node_.parameters_=node.parameters_;
			//SB::node_ = node;


			//NB the node is a proxy class does not hold the data

			//test.printOut();
			Node<i_HyperplaneNode> node(ndimensions, SB::t_data, SB::p_data);

			//save the paramenters
			for (int i = 0; i < ndimensions; i++)
				node.weights()[i] = test.weights[i];

			node.intercept() = test.intercept;

			//save the columns the first is how many

			for (int i = 0; i < ndimensions; i++)
			{
				node.columns_begin()[i] = test.columns[i];

			}


			SB::node_ = node;

			return i_HyperplaneNode;
		}
		else
		{
			return this->makeTerminalNode(features, labels, region, randint);

		}

	}

	/*
	 template<class T, class C, class Region>
	 void evaluateTest(Node<i_HyperplaneNode>& temp_test,
	 std::vector<Vals>& val_set, Region& region,
	 MultiArrayView<2, T, C> features)
	 {

	 typename Region::IndexIterator it;

	 val_set.clear();
	 val_set.resize(region.end() - region.begin());

	 int j = 0;
	 for (it = region.begin(); it != region.end(); it++)
	 {
	 MultiArrayView<2, T, C> current_row = rowVector(features, *it);

	 double val = 0.0;
	 for (int i = 0; i < ndimensions; i++)
	 {
	 val += temp_test.weights()[i]
	 * current_row[temp_test.columns_begin()[i]];

	 }

	 val_set[j].value = val;

	 val_set[j].index = *it;
	 ++j;

	 }

	 }
	 */
	template<class T, class C, class Region>
	void evaluateTest(SplitParams& temp_test, std::vector<Vals>& val_set,
			Region& region, MultiArrayView<2, T, C> features)
	{

		typename Region::IndexIterator it;

		val_set.clear();
		val_set.resize(region.size());

		int j = 0;
		for (it = region.begin(); it != region.end(); it++)
		{
			MultiArrayView<2, T, C> current_row = rowVector(features, *it);

			double val = 0.0;
			for (int i = 0; i < ndimensions; i++)
			{
				val += temp_test.weights[i] * current_row[temp_test.columns[i]];

			}

			val_set[j].value = val;

			val_set[j].index = *it;
			++j;

		}

		//sort the auxiliary structure vector
		std::sort(val_set.begin(), val_set.end());
	}

	template<class Random>
	void generateTest(SplitParams& test, Random& randint)
	{
		for (int i = 0; i < ndimensions; i++)
		{

			test.columns[i] = randint(SB::ext_param_.column_count_);

			if ((randint(2) + 1) % 2)
			{
				test.weights[i] = 1;

			}
			else
				test.weights[i] = -1;
		}
	}

	template<class Region>
	void splitTheRegion(std::vector<Vals>& valSet, Region& region,
			Region& temp_left, Region& temp_right, int tr)
	{
		int index = -1;

		//Find the max index that val< tr
		for (int ii = 0; ii < region.size(); ++ii)
		{
			if( valSet[ii].value < tr)
			{
				index = ii;

			}
			region[ii] = valSet[ii].index;
		}
		++index;

		temp_left.setRange(region.begin(), region.begin() + index);

		temp_right.setRange(region.begin() + index, region.end());
	}

	template<class Region, class T2, class C2>
	double measureSplit(Region& temp_left, Region& temp_right, MultiArrayView<
			2, T2, C2> labels)
	{
		typedef MultiArrayView<2, T2, C2> label_type;
		double l_size = temp_left.size();
		double r_size = temp_right.size();
		//std::cout << "The mode is " << mode << std::endl;

		if (mode == 0)
		{
			typename LossTraits<CriterionFunctor1, label_type>::type left(
					columnVector(labels, 0), SB::ext_param_); //initialize left and right region
			typename LossTraits<CriterionFunctor1, label_type>::type right(
					columnVector(labels, 0), SB::ext_param_);



			//temp_right.printRange();

			//the entropy is computed as sum -pi*log(pi)
			double left_gini = left.increment(temp_left.begin(),
					temp_left.end());

			//std::cout << "The left gini is " << left_gini << std::endl;
			//std::cout << "left region size is" << temp_left.size() << std::endl;
			//temp_left.printRange();


			double right_gini = right.increment(temp_right.begin(),
					temp_right.end());

			//std::cout << "The right gini is " << right_gini << std::endl;
			//std::cout << "right region size is" << temp_right.size()
			//		<< std::endl;

			//We want to maximize the info gain thus the wegted sum of the
			//negative entropy of the children should be as big as possible
			return  - (left_gini  + right_gini ) / (l_size
						+ r_size);
			//return left_gini+right_gini;
		}
		else
		{
			typename LossTraits<CriterionFunctor2, label_type>::type left(
					labels, SB::ext_param_);
			typename LossTraits<CriterionFunctor2, label_type>::type right(
					labels, SB::ext_param_);

			double left_gini = left.increment(temp_left.begin(),
					temp_left.end());

			double right_gini = right.increment(temp_right.begin(),
					temp_right.end());

			//std::cout << "The left variance is " << left_gini << std::endl;
			//std::cout << "The right variance  is " << right_gini << std::endl;

			return - (left_gini + right_gini ) / (l_size
					+ r_size);
			//return left_gini+right_gini;

		}

	}

	template<class T, class C, class T2, class C2, class Region, class Random>
	bool optimizeTest(MultiArrayView<2, T, C> features, MultiArrayView<2, T2,
	//C2> labels, Node<i_HyperplaneNode>& test, Region & region,
			C2> labels, SplitParams& test, Region & region,
			ArrayVector<Region>& childRegions, Random & randint)
	{

		bool found = false;

		//Maximize a certain measure
		double best_measure = -DBL_MAX;

		double temp_measure ;

		SplitParams bestTest(ndimensions);


		//ArrayVector<int> top_;
		//ArrayVector<double> param_;
		//Node<i_HyperplaneNode> temp_test(ndimensions, top_, param_);

		//SplitParams temp_test(ndimensions);

		std::vector<Vals> valSet;
		std::vector<Vals> bestValSet;


		//Find the best test of nsplits2try iterations
		for (int i = 0; i < nsplits2try; i++)
		{

			Region temp_left;
			Region temp_right;


			//generate Test without threshold
			//this->generateTest(temp_test, randint);

			this->generateTest(test, randint);


			//evaluate the test
			//this->evaluateTest(temp_test, valSet, region, features);
			this->evaluateTest(test, valSet, region, features);

			/*
			 for (int i=0;i<valSet.size();i++)
			 {
			 std::cerr << valSet[i].value << ", " ;
			 }
			*/


			const Vals& min_val = valSet.front();
			const Vals& max_val = valSet.back();

			double d;

			d = max_val.value - min_val.value;

			if (d > 0)
			{
				for ( unsigned int j = 1; j < 10; ++j)
				{
					//FIXME make more generic threshold
					if (d < 1)
						throw std::runtime_error("too small");

					int tr = randint(d) + min_val.value;

					//temp_test.intercept = tr;

					//Remember that val set is sorted!
					splitTheRegion(valSet, region, temp_left, temp_right, tr);

					//Measure the quality of the split the big it is the better
					temp_measure = measureSplit(temp_left, temp_right, labels);

					//std::cout << "tem_measure " << temp_measure << std::endl;
					//std::cout << "best_measure " << best_measure << std::endl;

					//std::cerr << "temp test parameters size " << temp_test.parameter_size_ << std::endl;

					//do not allow empty set splits

					if (temp_left.size() > 0 && temp_right.size() > 0)
					{
						if (temp_measure > best_measure)
						{
							found = true;
							test.intercept = tr;

							bestTest=test;

							bestValSet=valSet;

							//std::cerr << "found better intercept " << tr << std:: endl;

							/*
							std::cerr <<"valSet best split values "<< std::endl;

							for(int yy=0; yy<valSet.size();yy++)
							{
								std::cerr<< valSet[yy].value<< ", " ;
							}

							std::cerr<<std::endl;

							std::cerr <<"valSet best split indexes "<< std::endl;

														for(int yy=0; yy<valSet.size();yy++)
														{
															std::cerr<< valSet[yy].index<< ", " ;
														}

														std::cerr<<std::endl;

							*/
							//test.printOut();

							best_measure = temp_measure;
							//childRegions[0] = temp_left;
							//childRegions[1] = temp_right;

							//This is to not too crash later on


							//test.topology_=temp_test.topology_;
							//test.parameters_=temp_test.parameters_;


							//std::cerr << "intercept " << tr << std::endl;


						}

					}
				} //end for j
			}
		}//end iter

		if (found==true){


		//std::cout << "found " << found << std::endl;
		test=bestTest;
		splitTheRegion(bestValSet, region, childRegions[0], childRegions[1], bestTest.intercept);

		childRegions[0].classCounts().resize(
				region.classCounts().size());
		childRegions[1].classCounts().resize(
				region.classCounts().size());
		childRegions[0].classCountsIsValid = true;
		childRegions[1].classCountsIsValid = true;


		}

		return found;

	}

};


typedef RandomHyperplaneSplit<GiniCriterion, SpecialLSQLoss, HoughTag>
		HoughSplitRandomGini;

typedef RandomHyperplaneSplit<EntropyCriterion, SpecialLSQLoss, HoughTag>
		HoughSplitRandomEntropy;




template<class ColumnDecisionFunctor1, class ColumnDecisionFunctor2,
		class Tag = HoughTag>
class HoughThresholdSplit: public SplitBase<Tag>
{
public:

	typedef SplitBase<Tag> SB;

	ArrayVector<Int32> splitColumns;
	ColumnDecisionFunctor1 bgfunc1;
	ColumnDecisionFunctor2 bgfunc2; //regression functor

	double region_gini_;
	ArrayVector<double> min_gini_;
	ArrayVector<ptrdiff_t> min_indices_;
	ArrayVector<double> min_thresholds_;

	int bestSplitIndex;

	int splitmode;

	double minGini() const
	{
		return min_gini_[bestSplitIndex];
	}
	int bestSplitColumn() const
	{
		return splitColumns[bestSplitIndex];
	}
	double bestSplitThreshold() const
	{
		return min_thresholds_[bestSplitIndex];
	}

	template<class T>
	void set_external_parameters(ProblemSpec<T> const & in)
	{
		SB::set_external_parameters(in);
		bgfunc1.set_external_parameters(SB::ext_param_);
		bgfunc2.set_external_parameters(SB::ext_param_);
		int featureCount_ = SB::ext_param_.column_count_;
		splitColumns.resize(featureCount_);
		for (int k = 0; k < featureCount_; ++k)
			splitColumns[k] = k;
		min_gini_.resize(featureCount_);
		min_indices_.resize(featureCount_);
		min_thresholds_.resize(featureCount_);
	}

	template<class T, class C, class T2, class C2, class Region, class Random>
	int makeTerminalNode(MultiArrayView<2, T, C> features, MultiArrayView<2,
			T2, C2> labels, Region & region, Random randint)
	{
		Node<e_HoughTerminalNode> ret(SB::t_data, SB::p_data);
		SB::node_ = ret;

		MultiArrayView<2, T, C> labels1;
		//FIXME save the variance of the column except first
		typename MultiArrayView<2, T, C>::difference_type begin(0, 1);
		typename MultiArrayView<2, T, C>::difference_type end(labels.shape(0),
				labels.shape(1));

		labels1 = labels.subarray(begin, end);

		/*std::cout << "here" << std::endl;
		 std::cout << labels1.shape(0)<< " " << labels1.shape(1) << std::endl;

		 typename Region::IndexIterator iter=region.begin();

		 while(iter!=region.end()){
		 for(int ii = 0; ii < 2; ++ii)
		 {
		 std::cout  << labels1(*iter, ii) << " " ;

		 }
		 iter++;}
		 */

		//std::cout<< "the variance to be stored is " <<  bgfunc2.loss_of_region(labels1,
		//        region.begin(),
		//        region.end(),
		//        region.classCounts()) << std::endl;

		//region.printRange();
		//region.printClassCounts();


		//cout<<" here";

		if (SB::ext_param_.class_weights_.size() != region.classCounts().size())
		{
			std::copy(region.classCounts().begin(), region.classCounts().end(),
					ret.prob_begin());
		}
		else
		{
			std::transform(region.classCounts().begin(),
					region.classCounts().end(),
					SB::ext_param_.class_weights_.begin(), ret.prob_begin(),
					std::multiplies<double>());
		}
		detail::Normalise<ClassificationTag>::exec(ret.prob_begin(),
				ret.prob_end());
		//std::copy(ret.prob_begin(), ret.prob_end(), std::ostream_iterator<double>(std::cerr, ", " ));
		//std::cerr << std::endl;

		ret.weights() = region.size();
		return e_HoughTerminalNode;
	}

	template<class T, class C, class T2, class C2, class Region, class Random>
	int findBestSplit(MultiArrayView<2, T, C> features, MultiArrayView<2, T2,
			C2> labels, Region & region, ArrayVector<Region>& childRegions,
			Random & randint)
	{

		//FIXME: workouround for BIG bug
		MultiArrayView<2,T2,C2> integer_labels=columnVector(labels, 0);
		if (std::accumulate(region.classCounts().begin(),
				region.classCounts().end(), 0) != region.size())
		{
			RandomForestClassCounter<MultiArrayView<2, T2, C2> , ArrayVector<
					double> > counter(integer_labels,
					region.classCounts());
			std::for_each(region.begin(), region.end(), counter);
			region.classCountsIsValid = true;
		}

		splitmode = (randint(2) + 1) % 2;

		if (region.classCounts()[0] + region.classCounts()[1] != region.size())
			throw std::runtime_error("How can it be?");

		if (region.classCounts()[0] / double(region.size()) <= 0.05)
		{
			//std::cout << "to few examples to choose entropy" << std::endl;
			splitmode = 1;
		}

		if (splitmode == 0)
		{
			//std::cout << "minimize the Gini index" << std::endl;
			return findBestSplitClassification(features, labels, region,
					childRegions, randint);
		}
		else
		{
			//std::cout << "minimize the Variance" << std::endl;
			return findBestSplitRegression(features, labels, region,
					childRegions, randint);
		}

	}

	//FIXME this is actually done through an hack I create a
	//view on the original label matrix to make the classification act only
	//on the last columns

	template<class T, class C, class T2, class C2, class Region, class Random>
	int findBestSplitClassification(MultiArrayView<2, T, C>& features,
			MultiArrayView<2, T2, C2>& labels, Region & region, ArrayVector<
					Region>& childRegions, Random & randint)
	{

		typedef typename Region::IndexIterator IndexIterator;
		if (region.size() == 0)
		{
			std::cerr
					<< "SplitFunctor::findBestSplit(): stackentry with 0 examples encountered\n"
						"continuing learning process....";
		}

		//region.printClassCounts();

		//detail::Correction2::exec(region,columnVector(labels,0));


		// Is the region pure already?
		region_gini_ = bgfunc1.loss_of_region(columnVector(labels, 0),
				region.begin(), region.end(), region.classCounts());

		//if(region_gini_ <= SB::ext_param_.precision_)
		//    return  makeTerminalNode(features, labels, region, randint);

		// select columns  to be tried.
		for (int ii = 0; ii < SB::ext_param_.actual_mtry_; ++ii)
		{

			//DEBUG information
			//std::cout << "the shape is: " << features.shape(1) << std::endl;
			//std::cout << "split the cloumn " << splitColumns[ii] << std::endl;

			for (int i = 0; i < splitColumns.size(); ++i)
			{
				//std::cout << splitColumns[i] << std::endl;
			}
			std::swap(splitColumns[ii], splitColumns[ii + randint(
					features.shape(1) - ii)]);
		}

		// find the best gini index
		bestSplitIndex = 0;
		double current_min_gini = region_gini_;
		int num2try = features.shape(1);
		for (int k = 0; k < num2try; ++k)
		{
			//this functor does all the work  it pass a view on the column of the original
			//feature matrix to the underlying functor
			bgfunc1(columnVector(features, splitColumns[k]), columnVector(
					labels, 0), region.begin(), region.end(),
					region.classCounts());
			min_gini_[k] = bgfunc1.min_gini_;
			min_indices_[k] = bgfunc1.min_index_;
			min_thresholds_[k] = bgfunc1.min_threshold_;
#ifdef CLASSIFIER_TEST
			if( bgfunc1.min_gini_ < current_min_gini
					&& !closeAtTolerance(bgfunc1.min_gini_, current_min_gini))
#else
			if (bgfunc1.min_gini_ < current_min_gini)
#endif
			{
				current_min_gini = bgfunc1.min_gini_;
				childRegions[0].classCounts() = bgfunc1.bestCurrentCounts[0];
				childRegions[1].classCounts() = bgfunc1.bestCurrentCounts[1];
				childRegions[0].classCountsIsValid = true;
				childRegions[1].classCountsIsValid = true;

				bestSplitIndex = k;
				num2try = SB::ext_param_.actual_mtry_;
			}
		}
		//std::cerr << current_min_gini << "curr " << region_gini_ << std::endl;
		// did not find any suitable split

		//if(closeAtTolerance(current_min_gini, region_gini_))
		//    return  makeTerminalNode(features, labels, region, randint);

		//create a Node for output
		Node<i_ThresholdNode> node(SB::t_data, SB::p_data);
		SB::node_ = node;
		node.threshold() = min_thresholds_[bestSplitIndex];
		node.column() = splitColumns[bestSplitIndex];

		// partition the range according to the best dimension
		SortSamplesByDimensions<MultiArrayView<2, T, C> > sorter(features,
				node.column(), node.threshold());
		IndexIterator bestSplit = std::partition(region.begin(), region.end(),
				sorter);
		// Save the ranges of the child stack entries.
		childRegions[0].setRange(region.begin(), bestSplit);
		childRegions[0].rule = region.rule;
		childRegions[0].rule.push_back(std::make_pair(1, 1.0));
		childRegions[1].setRange(bestSplit, region.end());
		childRegions[1].rule = region.rule;
		childRegions[1].rule.push_back(std::make_pair(1, 1.0));

		//std::cout << "Split region A: " << childRegions[0].size()
		//		/ float(region.size()) << std::endl;
		//std::cout << "Split region B: " << childRegions[1].size()
		//		/ float(region.size()) << std::endl;
		//std::cout << "Size A/B " << childRegions[0].size() << ", "
		//		<< childRegions[1].size() << std::endl;

		return i_ThresholdNode;
	}

	//FIXME this is actually done through an hack I create a
	//view on the original label matrix to make the classification act only
	//on the last columns
	template<class T, class C, class T2, class C2, class Region, class Random>
	int findBestSplitRegression(MultiArrayView<2, T, C>& features,
			MultiArrayView<2, T2, C2>& labels, Region & region, ArrayVector<
					Region>& childRegions, Random & randint)
	{

		//Create a view only onto the regression labels: Not used anymore we need the other labels to decide for the variance
		//typename MultiArrayView<2, T, C>::difference_type Begin(0, 1);
		//typename MultiArrayView<2, T, C>::difference_type End(labels.shape(0),
		//		labels.shape(1));

		//labels=labels1.subarray(Begin,End);


		typedef typename Region::IndexIterator IndexIterator;
		if (region.size() == 0)
		{
			std::cerr
					<< "SplitFunctor::findBestSplit(): stackentry with 0 examples encountered\n"
						"continuing learning process....";
		}

		// Is the region pure already?
		region_gini_ = bgfunc2.loss_of_region(labels, region.begin(),
				region.end(), region.classCounts());

		//if(region_gini_ <= SB::ext_param_.precision_)
		//    return  makeTerminalNode(features, labels, region, randint);

		// select columns  to be tried.
		for (int ii = 0; ii < SB::ext_param_.actual_mtry_; ++ii)
		{

			//DEBUG information
			//std::cout << "the shape is: " << features.shape(1) << std::endl;
			//std::cout << "split the cloumn " << splitColumns[ii] << std::endl;

			for (int i = 0; i < splitColumns.size(); ++i)
			{
				//std::cout << splitColumns[i] << std::endl;
			}
			std::swap(splitColumns[ii], splitColumns[ii + randint(
					features.shape(1) - ii)]);
		}

		// find the best gini index
		bestSplitIndex = 0;
		double current_min_gini = region_gini_;
		int num2try = features.shape(1);
		for (int k = 0; k < num2try; ++k)
		{
			//this functor does all the work  it pass a view on the column of the original
			//feature matrix to the underlying functor
			bgfunc2(columnVector(features, splitColumns[k]), labels,
					region.begin(), region.end(), region.classCounts());
			min_gini_[k] = bgfunc2.min_gini_;
			min_indices_[k] = bgfunc2.min_index_;
			min_thresholds_[k] = bgfunc2.min_threshold_;
#ifdef CLASSIFIER_TEST
			if( bgfunc2.min_gini_ < current_min_gini
					&& !closeAtTolerance(bgfunc2.min_gini_, current_min_gini))
#else
			if (bgfunc2.min_gini_ < current_min_gini)
#endif
			{
				current_min_gini = bgfunc2.min_gini_;
				//childRegions[0].classCounts() = bgfunc2.bestCurrentCounts[0];
				//childRegions[1].classCounts() = bgfunc2.bestCurrentCounts[1];
				//childRegions[0].classCountsIsValid = true;
				//childRegions[1].classCountsIsValid = true;

				bestSplitIndex = k;
				num2try = SB::ext_param_.actual_mtry_;
			}
		}
		//std::cerr << current_min_gini << "curr " << region_gini_ << std::endl;
		// did not find any suitable split
		if (closeAtTolerance(current_min_gini, region_gini_))
			return makeTerminalNode(features, labels, region, randint);

		//create a Node for output
		Node<i_ThresholdNode> node(SB::t_data, SB::p_data);
		SB::node_ = node;
		node.threshold() = min_thresholds_[bestSplitIndex];
		node.column() = splitColumns[bestSplitIndex];

		// partition the range according to the best dimension
		SortSamplesByDimensions<MultiArrayView<2, T, C> > sorter(features,
				node.column(), node.threshold());
		IndexIterator bestSplit = std::partition(region.begin(), region.end(),
				sorter);
		// Save the ranges of the child stack entries.
		childRegions[0].setRange(region.begin(), bestSplit);
		childRegions[0].rule = region.rule;
		childRegions[0].rule.push_back(std::make_pair(1, 1.0));
		childRegions[1].setRange(bestSplit, region.end());
		childRegions[1].rule = region.rule;
		childRegions[1].rule.push_back(std::make_pair(1, 1.0));

		//std::cout << "Split region A: " << childRegions[0].size()
		//		/ float(region.size()) << std::endl;
		//std::cout << "Split region B: " << childRegions[1].size()
		//		/ float(region.size()) << std::endl;
		//std::cout << "Size A/B " << childRegions[0].size() << ", "
		//		<< childRegions[1].size() << std::endl;

		return i_ThresholdNode;
	}
};


typedef HoughThresholdSplit<BestGiniOfColumn<EntropyCriterion> ,
		BestGiniOfColumn<SpecialLSQLoss> , HoughTag> HoughSplitEntropy;

typedef HoughThresholdSplit<BestGiniOfColumn<GiniCriterion> ,
		BestGiniOfColumn<SpecialLSQLoss> , HoughTag> HoughSplitGini;


}
}

}
#endif
