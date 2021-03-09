#include <gtest/gtest.h>

#include <iostream>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder.hpp>
#include <memory>
#include <random>

using namespace umpalumpa::extrema_finder;
using namespace umpalumpa::data;

template <typename T>
void GenerateData(std::unique_ptr<T>& data, size_t elems) {
    auto mt = std::mt19937(42);
    auto dist = std::normal_distribution<float>((float)0, (float)1);
    for (size_t i = 0; i < elems; ++i) {
        data[i] = dist(mt);
    }
}

template <typename T>
void PrintData(std::unique_ptr<T>& data, const Size size) {
    for (size_t n = 0; n < size.n; ++n) {
        size_t offset = n * size.single;
        for (size_t i = 0; i < size.single; ++i) {
            printf("%+.3f\t", data[offset + i]);
        }
        std::cout << "\n";
    }
}

TEST(ExtermaFinder, basic1) {
    auto sizeData = Size(10, 1, 1, 3);
    auto settings = Settings(SearchType::kMax, SearchLocation::kEntire, SearchResult::kValue, sizeData, 10);
    auto data = std::unique_ptr<float[]>(new float[sizeData.total]);

    GenerateData(data, sizeData.total);
    PrintData(data, sizeData);

    auto sizeValues = Size(sizeData.n, 1, 1, 1);
    auto values = std::unique_ptr<float[]>(new float[sizeValues.total]);
    auto valuesP = Payload<float>(values.get(), {sizeValues, sizeValues}, {});

    auto dataP = SearchData<float>(data.get(), {sizeData, sizeData}, {});
    // }, nullptr))

    auto searcher = SingleExtremaFinder<float>();

    ASSERT_TRUE(searcher.execute({&valuesP, nullptr}, dataP, settings, false));
}