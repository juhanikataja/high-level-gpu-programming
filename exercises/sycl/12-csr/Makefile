default: build

# Set compiler
KOKKOS_PATH = $(shell pwd)/../kokkos
# CXX = hipcc
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper

# Variables for the Makefile.kokkos
# KOKKOS_DEVICES = "HIP"
KOKKOS_DEVICES = "Cuda"
#KOKKOS_ARCH = "VEGA90A"
KOKKOS_ARCH = "Volta70"

# Include Makefile.kokkos
include $(KOKKOS_PATH)/Makefile.kokkos

matvec_k: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) matvec_k.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(KOKKOS_LDFLAGS) matvec_k.cpp $(KOKKOS_LIBS) -o matvec_k
