#//////////////////////////////////////////////////////////////
#//   ____                                                   //
#//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
#//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
#//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
#//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
#//                             |_|             |_|          //
#//////////////////////////////////////////////////////////////
#//                                                          //
#//  Script, 2022                                            //
#//  Created: 19, April, 2022                                //
#//  Modified: 19, April, 2022                               //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

CXX_STANDARD := 17

.PHONY: all release debug minsizerel relwithdebinfo minsizerel relwithdebinfo release-clang debug-clang

all: release debug minsizerel relwithdebinfo minsizerel relwithdebinfo release-clang debug-clang

release:
	cmake -B build/$@ -S krackx -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=$(CXX_STANDARD)
	ninja -C build/$@
	ctest --verbose --parallel $(nproc) --test-dir build/$@

release-clang:
	cmake -B build/$@ -S krackx -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=$(CXX_STANDARD) \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	ninja -C build/$@
	ctest --verbose --parallel $(nproc) --test-dir build/$@

debug:
	cmake -B build/$@ -S krackx -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=$(CXX_STANDARD)
	ninja -C build/$@
	ctest --verbose --parallel $(nproc) --test-dir build/debug

debug-clang:
	cmake -B build/$@ -S krackx -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=$(CXX_STANDARD) \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
	ninja -C build/$@
	ctest --verbose --parallel $(nproc) --test-dir build/$@

minsizerel:
	cmake -B build/$@ -S krackx -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_CXX_STANDARD=$(CXX_STANDARD)
	ninja -C build/$@
	ctest --verbose --parallel $(nproc) --test-dir build/$@

relwithdebinfo:
	cmake -B build/$@ -S krackx -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_STANDARD=$(CXX_STANDARD)
	ninja -C build/$@
	ctest --verbose --parallel $(nproc) --test-dir build/$@


clean:
	rm -rf build
