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
 * Heap layout of the Scatter creation policy, rendered by
 * `python3 config.py render-param ScatterAlloc <config-name>` from the
 * `heap` object of the selected config.json config.
 */

#pragma once

namespace picongpu
{
    template<uint32_t T_pageSize, uint32_t T_accessBlockSize, uint32_t T_regionSize,
             uint32_t T_wasteFactor, bool T_resetFreedPages>
    struct ScatterHeapConfig
    {
        static constexpr auto pagesize = T_pageSize;
        static constexpr auto accessblocksize = T_accessBlockSize;
        static constexpr auto regionsize = T_regionSize;
        static constexpr auto wastefactor = T_wasteFactor;
        static constexpr auto resetfreedpages = T_resetFreedPages;
    };

} // namespace picongpu
