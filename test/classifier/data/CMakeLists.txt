
#       message(status "Current Source: ${CMAKE_CURRENT_SOURCE_DIR}") 
#       message(status "Current Bin: ${CMAKE_CURRENT_BINARY_DIR}") 
#       configure_file(${CMAKE_CURRENT_SOURCE_DIR}/oldClassifier.log 
#                      ${CMAKE_CURRENT_BINARY_DIR}/oldClassifier.log
#                      COPYONLY)
VIGRA_COPY_TEST_DATA(oldClassifier.log oldsetTest.log)
VIGRA_COPY_TEST_DATA(regression1D_noise.h5 regressionND_noise.h5)
VIGRA_COPY_TEST_DATA(HF_data_cells_train.h5 HF_data_cells_test.h5)
VIGRA_COPY_TEST_DATA(hf_cells_reference_OE.h5 hf_cells_reference_OG.h5)
VIGRA_COPY_TEST_DATA(hf_cells_reference_RE.h5 hf_cells_reference_RG.h5)
