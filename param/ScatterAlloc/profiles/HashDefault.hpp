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
 * The fork's default scatter hashing parameters, re-exposed as a dependent
 * template on the heap config so every hash profile shares the same shape
 * (`template<class T_HeapConfig>`); this one ignores the heap and uses the
 * fork's constants.
 */

#pragma once

namespace picongpu
{
    template<class /*T_HeapConfig*/>
    struct HashDefault
    {
        static constexpr auto hashingK = 38183u;
        static constexpr auto hashingDistMP = 17497u;
        static constexpr auto hashingDistWP = 1u;
        static constexpr auto hashingDistWPRel = 1u;
    };

} // namespace picongpu
