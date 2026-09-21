/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Carlchristian Eckert, Julian Lenz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.

/** @file
 *
 * The fork's default FlatterScatter hash config, re-exposed as a dependent
 * template on the heap config so the FlatterScatter hash profiles share
 * the same shape as the Scatter ones. The policy calls the hash as
 * `hash<T_HeapConfig::pagesize>(acc, bytes)`, and the template slot
 * carries the heap config as well.
 */

#pragma once

#include <mallocMC/creationPolicies/FlatterScatter.hpp>

namespace picongpu
{
    template<class /*T_HeapConfig*/>
    struct FsHashDefault
    {
        using Upstream = mallocMC::CreationPolicies::FlatterScatterAlloc::DefaultFlatterScatterHashConfig;
        static constexpr auto blockStride = Upstream::blockStride;
        ALPAKA_FN_INLINE ALPAKA_FN_ACC
        template<uint32_t T_PageSize, class TAcc>
        static auto hash(TAcc const& acc, uint32_t const numBytes) -> uint32_t
        {
            return Upstream::template hash<T_PageSize>(acc, numBytes);
        }
    };

} // namespace picongpu
